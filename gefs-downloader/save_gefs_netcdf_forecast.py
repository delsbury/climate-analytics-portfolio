#!/usr/bin/env python3
"""
Script to access NOAA GEFS forecast data and save multiple variables for all 
ensemble members concatenated along member and forecast hour dimensions to netCDF files.

OPTIMIZED VERSION (v4) - WITH PRIORITY 1 PERFORMANCE OPTIMIZATIONS:
- Processes variables sequentially (one at a time) to minimize peak memory
- Immediate deletion of raw data when starting each variable
- Optimized garbage collection (reduced frequency for better performance)
- Optimized chunk size (15) for better balance of memory vs speed
- Reduced compression level (complevel=2) for faster netCDF writing
- Optimized download parallelism (reduced workers to balance speed vs memory)

Processes forecasts every 6 hours from 6 to 240 hours (6, 12, 18, ..., 240).
For each variable, downloads all ensemble members for all forecast hours and saves
to a single netCDF file with all members and forecast hours concatenated.

Extracts the following variables (one file per variable):
- 2m temperature - TMP at 2 m above ground
- 10 meter u-wind (zonal wind) - UGRD at 10 m above ground
- Mean sea level pressure - PRMSL at mean sea level
- Total precipitation - APCP at surface (0-3 hour accumulation)
- Downward short wave radiation flux - DSWRF at surface (0-3 hour average)

Uses parallel processing to request multiple ensemble members simultaneously for faster downloads.
Uses quarter degree resolution (0.25 degree) GEFS data (product="atmos.25").

Uses the Herbie package to access GEFS data:
https://herbie.readthedocs.io/

GEFS has 31 ensemble members: member 0 (control) and members 1-30 (perturbed).
"""

from herbie import Herbie
import xarray as xr
import numpy as np
from datetime import datetime, timezone
import time
import ssl
import os
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure SSL to handle certificate issues for Cartopy downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Check for netCDF4 availability
try:
    import netCDF4
except ImportError:
    raise ImportError(
        "netCDF4 is required for this script. Install it with: pip install netCDF4"
    )


# Define variables to extract with their Herbie request strings
VARIABLES = {
    't2m': {
        'description': '2m temperature',
        'herbie_request': 'TMP:2 m',
        'search_patterns': ['t2m', 'tmp', 'temperature', 'tmp2m', 'tmp_2m'],
        'level': '2 m above ground',
        'short_name': 'TMP',
        'file_prefix': 't2m'
    },
    'ugrd_10m': {
        'description': '10 meter u-wind (zonal wind)',
        'herbie_request': 'UGRD:10 m',
        'search_patterns': ['ugrd', 'u-component', 'u_wind', 'u-component_of_wind'],
        'level': '10 m above ground',
        'short_name': 'UGRD',
        'file_prefix': 'ugrd_10m'
    },
    'prmsl': {
        'description': 'Mean sea level pressure',
        'herbie_request': 'PRMSL:mean sea level',
        'search_patterns': ['prmsl', 'mslp', 'mean_sea_level_pressure', 'pressure_reduced_to_msl'],
        'level': 'mean sea level',
        'short_name': 'PRMSL',
        'file_prefix': 'prmsl'
    },
    'apcp': {
        'description': 'Total precipitation',
        'herbie_request': 'APCP:surface',
        'search_patterns': ['apcp', 'precipitation', 'total_precipitation', 'precip'],
        'level': 'surface',
        'short_name': 'APCP',
        'file_prefix': 'apcp',
        'requires_forecast': True  # Only available at f003 and later (not at f000)
    },
    'dswrf': {
        'description': 'Downward short wave radiation flux',
        'herbie_request': 'DSWRF:surface',
        'search_patterns': ['dswrf', 'downward_short_wave', 'downward_sw', 'sw_down'],
        'level': 'surface',
        'short_name': 'DSWRF',
        'file_prefix': 'dswrf',
        'requires_forecast': True  # Only available at f003 and later (not at f000)
    }
}


def find_variable(ds, var_config):
    """
    Find a variable in the dataset using search patterns.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        GEFS dataset
    var_config : dict
        Variable configuration with search_patterns
    
    Returns:
    --------
    str or None
        Variable name if found, None otherwise
    """
    available_vars = list(ds.variables.keys())
    
    # Search for variable using patterns
    for pattern in var_config['search_patterns']:
        for var_name in available_vars:
            if pattern.lower() in var_name.lower():
                return var_name
    
    # Also check data variables (not just coordinates)
    data_vars = list(ds.data_vars.keys())
    for pattern in var_config['search_patterns']:
        for var_name in data_vars:
            if pattern.lower() in var_name.lower():
                return var_name
    
    return None


def extract_variable_data(ds, var_config, member, forecast_hour):
    """
    Extract and prepare a variable from a dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        GEFS dataset
    var_config : dict
        Variable configuration
    member : int
        Ensemble member number
    forecast_hour : int
        Forecast hour
    
    Returns:
    --------
    tuple: (xarray.DataArray, str) or (None, None)
        Variable data array and variable name, or (None, None) if not found
    """
    # Find the variable in the dataset
    var_name = find_variable(ds, var_config)
    
    if var_name is None:
        # If not found by search, try to get the first data variable
        data_vars = list(ds.data_vars.keys())
        if len(data_vars) > 0:
            var_name = data_vars[0]
        else:
            return None, None
    
    var_data = ds[var_name]
    
    # Handle different data shapes
    if len(var_data.shape) == 2:
        plot_data = var_data
    elif len(var_data.shape) == 3:
        # For 3D data, might need to select a level or time
        first_dim = var_data.dims[0]
        if first_dim in ['level', 'isobaric', 'height']:
            plot_data = var_data.isel({first_dim: 0})
        else:
            plot_data = var_data.isel({first_dim: 0})
    else:
        plot_data = var_data.squeeze()
    
    # Convert to float32 to save space
    if plot_data.dtype == 'float64':
        plot_data = plot_data.astype('float32')
    
    # Drop 'number' coordinate if present (it conflicts when concatenating members)
    # We'll use our own 'member' coordinate instead
    if 'number' in plot_data.coords:
        plot_data = plot_data.drop_vars('number')
    if 'number' in plot_data.dims:
        plot_data = plot_data.drop_dims('number')
    
    # Add member coordinate if not already present
    if 'member' not in plot_data.dims:
        plot_data = plot_data.expand_dims('member')
        plot_data = plot_data.assign_coords(member=[member])
    else:
        plot_data = plot_data.assign_coords(member=member)
    
    # Add forecast_hour as a coordinate/dimension
    if 'forecast_hour' not in plot_data.dims:
        plot_data = plot_data.expand_dims('forecast_hour')
        plot_data = plot_data.assign_coords(forecast_hour=[forecast_hour])
    else:
        plot_data = plot_data.assign_coords(forecast_hour=forecast_hour)
    
    return plot_data, var_name


def cleanup_grib_file(local_file):
    """
    Clean up downloaded GRIB file and any associated index files.
    
    Parameters:
    -----------
    local_file : str or Path
        Path to the GRIB file to delete
    """
    if local_file is None:
        return
    
    try:
        file_path = str(local_file)
        
        if os.path.exists(file_path):
            os.remove(file_path)
        
        idx_file = file_path + '.idx'
        if os.path.exists(idx_file):
            os.remove(idx_file)
            
    except Exception:
        pass  # Don't fail if cleanup fails


def download_and_extract_all_variables(args):
    """
    Download a GRIB file once and extract all variables from it.
    This is much more efficient than downloading the same file 5 times.
    
    Parameters:
    -----------
    args : tuple
        (date_str, cycle_str, member, forecast_hour, VARIABLES dict)
    
    Returns:
    --------
    tuple: (int, int, dict, dict)
        (member, forecast_hour, extracted_vars_dict, timing_info)
        extracted_vars_dict: {var_key: (data, var_name)}
    """
    date_str, cycle_str, member, forecast_hour, variables_dict = args
    
    timing_info = {
        'download_time': 0,
        'extract_times': {},
        'success': False
    }
    
    local_file = None
    herbie_obj = None
    
    try:
        # Download the file once
        t0 = time.time()
        date_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {cycle_str}:00"
        
        herbie_obj = Herbie(
            date=date_time,
            model="gefs",
            product="atmos.25",
            fxx=forecast_hour,
            member=member,
            priority=["aws", "google", "nomads", "azure", "pando", "pando2"]
        )
        
        local_file = herbie_obj.download()
        
        if local_file is None:
            raise RuntimeError(f"Failed to download GEFS data for member {member}, f{forecast_hour:03d}")
        
        timing_info['download_time'] = time.time() - t0
        
        # Extract all variables from the same file
        extracted_vars = {}
        
        for var_key, var_config in variables_dict.items():
            # Check if variable requires forecast hour > 0 (APCP and DSWRF only available at f003+)
            requires_forecast = var_config.get('requires_forecast', False)
            if requires_forecast and forecast_hour == 0:
                # Skip variables that aren't available at f000
                continue
            
            try:
                t_extract = time.time()
                
                # Open with xarray, requesting specific variable
                ds_result = herbie_obj.xarray(var_config['herbie_request'])
                
                # Handle case where xarray returns a list of datasets
                if isinstance(ds_result, list):
                    ds = None
                    for d in ds_result:
                        if len(d.data_vars) > 0:
                            ds = d
                            break
                    if ds is None:
                        ds = ds_result[0]
                else:
                    ds = ds_result
                
                if ds is None:
                    continue
                
                # Extract the variable data
                plot_data, var_name = extract_variable_data(ds, var_config, member, forecast_hour)
                
                if plot_data is not None:
                    # Load data into memory to detach from source dataset
                    plot_data = plot_data.load()
                    extracted_vars[var_key] = (plot_data, var_name)
                    timing_info['extract_times'][var_key] = time.time() - t_extract
                
                ds.close()
                
            except Exception:
                # Continue with other variables if one fails
                timing_info['extract_times'][var_key] = None
                continue
        
        timing_info['success'] = len(extracted_vars) > 0
        
        # Clean up the downloaded file
        cleanup_grib_file(local_file)
        
        return member, forecast_hour, extracted_vars, timing_info
        
    except Exception as e:
        timing_info['success'] = False
        timing_info['error'] = str(e)
        cleanup_grib_file(local_file)
        return member, forecast_hour, {}, timing_info


def download_all_data_once(date_str, cycle_str, ensemble_members, forecast_hours, max_workers=20):
    """
    Download all (member, forecast_hour) combinations once and extract all variables.
    This is much more efficient than downloading the same file 5 times per variable.
    
    OPTIMIZED: Reduced max_workers from 25 to 20 to reduce memory during download phase.
    
    Parameters:
    -----------
    date_str : str
        Date string in format YYYYMMDD
    cycle_str : str
        Cycle string in format HH
    ensemble_members : list
        List of ensemble member numbers
    forecast_hours : list
        List of forecast hours
    max_workers : int
        Maximum number of parallel workers (reduced to 20 for memory optimization)
    
    Returns:
    --------
    dict: {var_key: {(member, forecast_hour): (data, var_name)}}
        Organized results by variable
    """
    print(f"\n{'=' * 60}")
    print(f"Downloading all data (single download per file, extract all variables)")
    print(f"{'=' * 60}")
    print(f"Members: {len(ensemble_members)}, Forecast hours: {len(forecast_hours)}")
    print(f"Total downloads: {len(ensemble_members) * len(forecast_hours)} (was {len(ensemble_members) * len(forecast_hours) * len(VARIABLES)})")
    print(f"Parallel workers: {max_workers}")
    
    start_time = time.time()
    timing_info = {
        'download_times': [],
        'extract_times': {var_key: [] for var_key in VARIABLES.keys()},
        'failed_downloads': []
    }
    
    # Prepare arguments for parallel processing
    args_list = [
        (date_str, cycle_str, member, fhour, VARIABLES)
        for member in ensemble_members
        for fhour in forecast_hours
    ]
    
    # Process in parallel - download once, extract all variables
    all_results = {var_key: {} for var_key in VARIABLES.keys()}
    print(f"\nProcessing {len(args_list)} downloads in parallel (max_workers={max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {
            executor.submit(download_and_extract_all_variables, args): args
            for args in args_list
        }
        
        completed = 0
        for future in as_completed(future_to_args):
            completed += 1
            args = future_to_args[future]
            member, forecast_hour = args[2], args[3]
            
            try:
                member_num, fhour, extracted_vars, member_timing = future.result()
                
                if member_timing['success']:
                    timing_info['download_times'].append(member_timing['download_time'])
                    
                    # Organize results by variable
                    for var_key, (plot_data, var_name) in extracted_vars.items():
                        key = (member_num, fhour)
                        all_results[var_key][key] = {
                            'data': plot_data,
                            'var_name': var_name
                        }
                        if var_key in member_timing['extract_times'] and member_timing['extract_times'][var_key] is not None:
                            timing_info['extract_times'][var_key].append(member_timing['extract_times'][var_key])
                    
                    if completed % 20 == 0:
                        print(f"  Progress: {completed}/{len(args_list)} completed")
                else:
                    timing_info['failed_downloads'].append((member_num, fhour))
                    print(f"  Member {member_num}, f{fhour:03d} failed: {member_timing.get('error', 'Unknown error')}")
                    
            except Exception as e:
                timing_info['failed_downloads'].append((member, forecast_hour))
                print(f"  Member {member}, f{forecast_hour:03d} exception: {e}")
    
    total_time = time.time() - start_time
    timing_info['total_time'] = total_time
    
    print(f"\nDownload phase complete: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"  Successful: {len(args_list) - len(timing_info['failed_downloads'])}/{len(args_list)}")
    print(f"  Failed: {len(timing_info['failed_downloads'])}")
    
    return all_results, timing_info


def concatenate_variable_data(var_key, var_config, results_dict, forecast_hours):
    """
    Concatenate data for a single variable across all members and forecast hours.
    
    OPTIMIZED: Uses smaller chunk sizes (10 instead of 15) and more aggressive cleanup.
    
    Parameters:
    -----------
    var_key : str
        Variable key
    var_config : dict
        Variable configuration
    results_dict : dict
        {(member, forecast_hour): {'data': ..., 'var_name': ...}}
    forecast_hours : list
        List of forecast hours (may include hours not in results_dict)
    
    Returns:
    --------
    tuple: (xarray.Dataset, list, float)
        Concatenated dataset, actual forecast hours in data, and concatenation time
    """
    if len(results_dict) == 0:
        raise RuntimeError(f"No data available for {var_key}")
    
    print(f"\nConcatenating {var_key} ({len(results_dict)} datasets)...")
    
    # First, concatenate along member dimension for each forecast hour
    print("  Step 1: Concatenating members for each forecast hour...")
    t0 = time.time()
    datasets_by_fhour = {}
    
    for (member, fhour), result in results_dict.items():
        if fhour not in datasets_by_fhour:
            datasets_by_fhour[fhour] = []
        datasets_by_fhour[fhour].append((member, result['data']))
    
    # Get actual forecast hours present in the data
    actual_forecast_hours = sorted(datasets_by_fhour.keys())
    
    # Concatenate members for each forecast hour
    concatenated_by_fhour = {}
    
    for fhour in actual_forecast_hours:
        member_data = sorted(datasets_by_fhour[fhour], key=lambda x: x[0])
        datasets = [data for _, data in member_data]
        
        # Concatenate along member dimension
        combined = xr.concat(datasets, dim='member', coords='minimal', compat='override')
        concatenated_by_fhour[fhour] = combined
        
        # Cleanup individual datasets immediately after concatenation
        del datasets
        del member_data
    
    step1_time = time.time() - t0
    print(f"  Step 1 complete: {step1_time:.2f}s")
    
    # Cleanup intermediate data from Step 1
    del datasets_by_fhour
    # GC removed here - let automatic GC handle minor cleanup
    
    # Second, concatenate along forecast_hour dimension (chunked to reduce memory)
    print("  Step 2: Concatenating forecast hours (chunked, size=15)...")
    t0 = time.time()
    
    # OPTIMIZED: Chunk size 15 for better balance of memory vs speed
    chunk_size = 15
    chunked_results = []
    
    # Get first dataset for coordinates before chunking
    first_ds_for_coords = concatenated_by_fhour[actual_forecast_hours[0]]
    
    for i in range(0, len(actual_forecast_hours), chunk_size):
        chunk_hours = actual_forecast_hours[i:i+chunk_size]
        chunk_datasets = [concatenated_by_fhour[fhour] for fhour in chunk_hours]
        chunk_combined = xr.concat(chunk_datasets, dim='forecast_hour', 
                                   coords='minimal', join='outer', compat='override')
        chunked_results.append(chunk_combined)
        
        # Cleanup intermediate arrays to free memory immediately
        del chunk_datasets
        # GC removed here - only keep GC after final chunk processing
    
    # Final concatenation of chunks
    final_combined = xr.concat(chunked_results, dim='forecast_hour', 
                             coords='minimal', join='outer', compat='override')
    
    step2_time = time.time() - t0
    print(f"  Step 2 complete: {step2_time:.2f}s")
    
    # Get first dataset for coordinates before cleanup
    first_ds = first_ds_for_coords
    
    # Cleanup chunked results and concatenated_by_fhour immediately
    del chunked_results
    del concatenated_by_fhour
    del first_ds_for_coords
    gc.collect()
    
    coords = {}
    for coord_name in ['latitude', 'longitude', 'lat', 'lon', 'time', 'step', 'valid_time']:
        if coord_name in first_ds.coords:
            coords[coord_name] = first_ds.coords[coord_name]
    
    # Add member and forecast_hour coordinates
    coords['member'] = final_combined.coords['member']
    coords['forecast_hour'] = final_combined.coords['forecast_hour']
    
    # Use standardized variable name
    final_var_name = var_key
    final_ds = xr.Dataset({final_var_name: final_combined}, coords=coords)
    
    # Final cleanup
    del first_ds
    # GC removed here - dataset will be cleaned up after use
    
    return final_ds, actual_forecast_hours, step1_time + step2_time


def save_variable_netcdf(ds, var_key, var_config, date_str, cycle_str, 
                         forecast_hours, output_dir='.'):
    """
    Save concatenated dataset to netCDF file.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Concatenated dataset with all members and forecast hours
    var_key : str
        Variable key
    var_config : dict
        Variable configuration
    date_str : str
        Date string for filename
    cycle_str : str
        Cycle string for filename
    forecast_hours : list
        List of forecast hours processed
    output_dir : str
        Directory to save output file
    
    Returns:
    --------
    tuple: (str, float, float)
        Output file path, file size in MB, and save time
    """
    var_name = var_key
    
    # Add attributes
    fhour_min = min(forecast_hours)
    fhour_max = max(forecast_hours)
    ds.attrs = {
        'source': 'NOAA GEFS',
        'variable': var_config['description'],
        'short_name': var_config['short_name'],
        'level': var_config['level'],
        'forecast_hours': f'f{fhour_min:03d} to f{fhour_max:03d}',
        'forecast_hour_min': fhour_min,
        'forecast_hour_max': fhour_max,
        'num_forecast_hours': len(forecast_hours),
        'ensemble_members': f'0-{len(ds.member)-1}',
        'num_members': len(ds.member),
        'created': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }
    
    # Add variable-specific attributes
    if var_name in ds.data_vars:
        ds[var_name].attrs.update({
            'description': var_config['description'],
            'short_name': var_config['short_name'],
            'level': var_config['level']
        })
    
    # Optimized encoding for netCDF
    var_data = ds[var_name]
    
    # Determine chunk sizes based on data shape
    if len(var_data.shape) >= 2:
        # For spatial dimensions (last 2), use reasonable chunk sizes
        spatial_chunks = tuple(min(500, s) for s in var_data.shape[-2:])
        
        # For member and forecast_hour dimensions, use smaller chunks
        if len(var_data.shape) > 2:
            extra_dims = len(var_data.shape) - 2
            extra_chunks = tuple(min(10, s) for s in var_data.shape[:extra_dims])
            chunksizes = extra_chunks + spatial_chunks
        else:
            chunksizes = spatial_chunks
    else:
        chunksizes = None
    
    encoding = {
        var_name: {
            'zlib': True,
            'complevel': 2,  # Reduced from 4 to 2 for faster compression (Priority 1 optimization)
            'shuffle': True,
            'dtype': 'float32',
        }
    }
    
    if chunksizes is not None and len(chunksizes) == len(var_data.shape):
        encoding[var_name]['chunksizes'] = chunksizes
    
    # Encode coordinates
    for coord_name in ds.coords:
        if coord_name in ['latitude', 'longitude', 'lat', 'lon']:
            encoding[coord_name] = {
                'zlib': True,
                'complevel': 4,
                'dtype': 'float32'
            }
        elif coord_name in ['member', 'forecast_hour']:
            coord_data = ds.coords[coord_name]
            try:
                if coord_data.dtype.kind in ['i', 'u']:
                    encoding[coord_name] = {
                        'dtype': 'int32'
                    }
                else:
                    encoding[coord_name] = {
                        'dtype': 'float32',
                        '_FillValue': np.nan
                    }
            except Exception:
                encoding[coord_name] = {
                    'dtype': 'float32',
                    '_FillValue': np.nan
                }
        elif coord_name == 'number':
            encoding[coord_name] = {
                'dtype': 'float32',
                '_FillValue': np.nan
            }
    
    # Create output filename (using _v3 suffix as requested)
    fhour_min = min(forecast_hours)
    fhour_max = max(forecast_hours)
    output_file = os.path.join(
        output_dir,
        f"gefs_{var_config['file_prefix']}_all_members_{date_str}_{cycle_str}z_f{fhour_min:03d}_to_f{fhour_max:03d}_v3.nc"
    )
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to netCDF
    print(f"Saving to {output_file}...")
    t0 = time.time()
    ds.to_netcdf(
        output_file,
        format='NETCDF4',
        encoding=encoding,
        engine='netcdf4'
    )
    save_time = time.time() - t0
    print(f"Save time: {save_time:.2f}s")
    
    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    return output_file, file_size_mb, save_time


def main():
    """
    Main function to download and concatenate all variables, members, and forecast hours.
    
    OPTIMIZED: Processes variables sequentially (one at a time) to minimize peak memory.
    """
    start_time = time.time()
    print("=" * 60)
    print("GEFS Multivariable Download and Concatenation (v4 - OPTIMIZED)")
    print("=" * 60)
    
    # Sample initialization: December 05, 2025, 00Z
    date_str = "20251205"
    cycle_str = "00"
    
    # GEFS forecasts from f000 to f240, every 6 hours
    forecast_hours = list(range(0, 241, 6))  # 0, 6, 12, 18, ..., 240
    
    # GEFS has 31 ensemble members: 0 (control) and 1-30 (perturbed)
    ensemble_members = list(range(31))  # 0 to 30
    
    # Output directory
    output_dir = "/Volumes/Extreme SSD/weather/gefs/v2"
    
    print(f"Date: {date_str}")
    print(f"Cycle: {cycle_str}Z")
    print(f"Forecast Hours: {len(forecast_hours)} hours ({forecast_hours[0]} to {forecast_hours[-1]}, every 6 hours)")
    print(f"Ensemble Members: {len(ensemble_members)} (0=control, 1-30=perturbed)")
    print(f"Variables: {len(VARIABLES)} ({', '.join([v['short_name'] for v in VARIABLES.values()])})")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    # Track results for all variables
    all_results = {}
    total_files_size = 0
    total_save_time = 0
    
    # Check which output files already exist
    fhour_min = min(forecast_hours)
    fhour_max = max(forecast_hours)
    existing_files = {}
    for var_key, var_config in VARIABLES.items():
        output_file = os.path.join(
            output_dir,
            f"gefs_{var_config['file_prefix']}_all_members_{date_str}_{cycle_str}z_f{fhour_min:03d}_to_f{fhour_max:03d}_v3.nc"
        )
        if os.path.exists(output_file):
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"\nOutput file already exists for {var_key}: {output_file}")
            print(f"  File size: {file_size_mb:.2f} MB - Skipping")
            all_results[var_key] = {
                'output_file': output_file,
                'file_size_mb': file_size_mb,
                'save_time': 0,
                'timing': None,
                'success': True,
                'skipped': True
            }
            total_files_size += file_size_mb
            existing_files[var_key] = True
        else:
            existing_files[var_key] = False
    
    # If all files exist, skip download phase
    if all(existing_files.values()):
        print("\nAll output files already exist. Skipping download phase.")
    else:
        # Download all data once (optimized approach - downloads each file once, extracts all variables)
        print("\n" + "=" * 60)
        print("PHASE 1: Downloading all data")
        print("=" * 60)
        all_downloaded_data, download_timing = download_all_data_once(
            date_str, cycle_str, ensemble_members, forecast_hours, max_workers=20
        )
        
        # OPTIMIZED: Process variables SEQUENTIALLY (one at a time) to minimize peak memory
        print(f"\n{'=' * 60}")
        print("PHASE 2: Processing variables sequentially (memory-optimized)")
        print(f"{'=' * 60}")
        
        variables_to_process = [
            (var_key, var_config) 
            for var_key, var_config in VARIABLES.items()
            if not existing_files.get(var_key, False)
        ]
        
        if variables_to_process:
            print(f"Processing {len(variables_to_process)} variables sequentially...")
            
            for var_idx, (var_key, var_config) in enumerate(variables_to_process, 1):
                print(f"\n{'=' * 60}")
                print(f"Processing variable {var_idx}/{len(variables_to_process)}: {var_key} ({var_config['description']})")
                print(f"{'=' * 60}")
                
                try:
                    # CRITICAL OPTIMIZATION: Extract raw data for this variable and delete immediately
                    # This frees ~5.3 GB of memory (1/5 of total raw data) before concatenation starts
                    if var_key not in all_downloaded_data:
                        print(f"  Warning: No data found for {var_key}, skipping...")
                        all_results[var_key] = {'success': False, 'error': 'No data available'}
                        continue
                    
                    var_data = all_downloaded_data[var_key]
                    
                    # Delete from all_downloaded_data immediately to free memory
                    del all_downloaded_data[var_key]
                    gc.collect()  # Keep this GC - frees 5.3 GB of raw data (critical cleanup)
                    print(f"  Memory freed: removed {var_key} from raw data dictionary")
                    
                    # Concatenate data for this variable
                    ds, actual_forecast_hours, concat_time = concatenate_variable_data(
                        var_key, var_config, var_data, forecast_hours
                    )
                    
                    # Cleanup raw data immediately after concatenation
                    del var_data
                    # GC removed here - let automatic GC handle cleanup
                    print(f"  Concatenation complete: {concat_time:.2f}s")
                    
                    # Save to netCDF immediately (don't accumulate datasets)
                    output_file, file_size_mb, save_time = save_variable_netcdf(
                        ds, var_key, var_config, date_str, cycle_str, actual_forecast_hours, output_dir
                    )
                    
                    result = {
                        'output_file': output_file,
                        'file_size_mb': file_size_mb,
                        'save_time': save_time,
                        'concat_time': concat_time,
                        'success': True
                    }
                    all_results[var_key] = result
                    
                    total_files_size += file_size_mb
                    total_save_time += save_time
                    
                    # Cleanup dataset immediately after saving
                    try:
                        if hasattr(ds, 'close'):
                            ds.close()
                    except Exception:
                        pass
                    
                    del ds
                    # GC removed here - dataset already deleted, let automatic GC handle
                    print(f"  Variable {var_key} complete. Memory cleaned up.")
                    
                except Exception as e:
                    print(f"  Error processing variable {var_key}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results[var_key] = {'success': False, 'error': str(e)}
                
                # Force cleanup after each variable
                gc.collect()
            
            # Final cleanup of any remaining downloaded data
            del all_downloaded_data
            gc.collect()
            print("\nAll variables processed. All raw data cleaned up.")
        
        # Add download timing to results
        for var_key in all_results:
            if not all_results[var_key].get('skipped', False):
                all_results[var_key]['download_timing'] = download_timing
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    
    successful_vars = [v for v in VARIABLES.keys() if all_results.get(v, {}).get('success', False) and not all_results.get(v, {}).get('skipped', False)]
    skipped_vars = [v for v in VARIABLES.keys() if all_results.get(v, {}).get('skipped', False)]
    failed_vars = [v for v in VARIABLES.keys() if not all_results.get(v, {}).get('success', False)]
    
    print(f"Total variables: {len(VARIABLES)}")
    print(f"Processed: {len(successful_vars)}")
    print(f"Skipped (already exist): {len(skipped_vars)}")
    print(f"Failed: {len(failed_vars)}")
    
    if skipped_vars:
        print(f"Skipped variables: {skipped_vars}")
    if failed_vars:
        print(f"Failed variables: {failed_vars}")
    
    if successful_vars or skipped_vars:
        print(f"\nTotal files size: {total_files_size:.2f} MB ({total_files_size/1024:.2f} GB)")
        if successful_vars:
            print(f"Total save time: {total_save_time:.2f}s ({total_save_time/60:.2f} min)")
        
        # Calculate average times
        if successful_vars:
            # Get download timing if available
            download_timing = None
            for var_key in successful_vars:
                if 'download_timing' in all_results[var_key]:
                    download_timing = all_results[var_key]['download_timing']
                    break
            
            if download_timing:
                print(f"\nDownload phase statistics:")
                if download_timing['download_times']:
                    print(f"  Average download time per file: {np.mean(download_timing['download_times']):.2f}s")
                for var_key in VARIABLES.keys():
                    if var_key in download_timing['extract_times'] and download_timing['extract_times'][var_key]:
                        print(f"  Average extract time for {var_key}: {np.mean(download_timing['extract_times'][var_key]):.2f}s")
            
            # Get concatenation and save times
            concat_times = []
            save_times = []
            for var_key in successful_vars:
                if 'concat_time' in all_results[var_key]:
                    concat_times.append(all_results[var_key]['concat_time'])
                if 'save_time' in all_results[var_key]:
                    save_times.append(all_results[var_key]['save_time'])
            
            if concat_times:
                print(f"\nProcessing phase statistics:")
                print(f"  Average concatenation time per variable: {np.mean(concat_times):.2f}s")
            if save_times:
                print(f"  Average save time per variable: {np.mean(save_times):.2f}s")
    
    total_elapsed = time.time() - start_time
    print(f"\nTotal runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min) / {total_elapsed/3600:.2f} hours")
    print("=" * 60)


if __name__ == "__main__":
    main()
