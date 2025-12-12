"""
SSW Wave Analysis - Switchable between Sea Level Pressure and Surface Temperature

Calculates and visualizes anomalies during Sudden Stratospheric Warmings (SSWs) 
using 15-day averaging periods. Includes statistical significance testing by 
comparing SSW composites to the same calendar dates in non-SSW years.

This version can seamlessly switch between:
- Sea level pressure (SLP) anomalies
- Surface air temperature anomalies

This version is optimized with:
- Constants at the top for easy configuration
- Modular functions for better maintainability
- Main block for script execution
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.colors as mcolors
import more_itertools as mit
import pandas as pd
from datetime import timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from palettable.colorbrewer.diverging import RdBu_11

# ============================================================================
# CONFIGURATION - Switch between SLP and Temperature
# ============================================================================

# Set to 'slp' for sea level pressure or 'temp' for surface temperature
VARIABLE_TYPE = 'slp'  # Change this to 'temp' for temperature analysis

# Variable-specific configurations
VAR_CONFIGS = {
    'slp': {
        'file_path': 'slp_1979-2009.nc',
        'var_name': 'slp',
        'unit_conversion': 100,  # Convert Pa to hPa
        'vmin': -6,
        'vmax': 6,
        'n_levels': 25,
        'colormap': RdBu_11.mpl_colormap.reversed(),
        'unit_label': 'hPa',
        'var_label': 'Sea level pressure',
        'output_file': 'ssw_slp_15day.png'
    },
    'temp': {
        'file_path': 'air_sfc-1979-2009.nc',
        'var_name': 'air',
        'unit_conversion': 1,  # No conversion needed
        'vmin': -4,
        'vmax': 4,
        'n_levels': 25,
        'colormap': RdBu_11.mpl_colormap.reversed(),
        'unit_label': 'K',
        'var_label': 'Surface air temperature',
        'output_file': 'ssw_temp_15day.png'
    }
}

# Get current configuration
CONFIG = VAR_CONFIGS[VARIABLE_TYPE]

# ============================================================================
# CONSTANTS
# ============================================================================

# File paths
U10_FILE = 'u10_60n-1979-2009.nc'

# Time range for SSW detection
TIME_START = '1979-11-01'
TIME_END = '2009-04-30'
SSW_MONTHS = [11, 12, 1, 2, 3]  # Months to search for SSWs

# SSW detection parameters
MIN_WESTERLY_DAYS = 10  # Minimum consecutive westerly days to avoid final warming
MIN_GROUP_SEPARATION = 20  # Minimum days between groups to be separate SSWs

# Composite analysis parameters
DAYS_BEFORE_SSW = 29  # Days before SSW to include
DAYS_AFTER_SSW = 30  # Days after SSW to include
TOTAL_DAYS = 60  # Total window: 29 before + 1 at event + 30 after

# 15-day averaging periods (indices in 60-day window)
PERIODS = [
    (0, 15),   # days 29->15: indices 0-14 (15 days before, ending 15 days before SSW)
    (15, 30),  # days 14->0: indices 15-29 (15 days before, ending at SSW)
    (30, 45),  # days 1->15: indices 30-44 (15 days after, starting 1 day after SSW)
    (45, 60)   # days 16->30: indices 45-59 (15 days after, starting 16 days after SSW)
]

PERIOD_LABELS = ['days -29 to -15', 'days -14 to 0', 
                 'days 1 to 15', 'days 16 to 30']

# Statistical significance parameters
ALPHA = 0.05  # Significance level (95% confidence interval)
N_BOOTSTRAP = 1000  # Number of bootstrap iterations

# Plotting parameters
FIG_SIZE = (10, 12)
FONT_NAME = 'Arial'
SUPTITLE_FONTSIZE = 20
TITLE_FONTSIZE = 20
CBAR_FONTSIZE = 12
DPI = 700

# Map settings
COASTLINE_LINEWIDTH = 1.5
COASTLINE_RESOLUTION = '50m'
COASTLINE_COLOR = 'slategray'
COASTLINE_ALPHA = 0.6
LAND_ALPHA = 0.3
MAP_EXTENT = [-180, 180, 10, 90]  # [lon_min, lon_max, lat_min, lat_max]
CIRCLE_POINTS = 200  # Points for circular boundary
HATCH_LINEWIDTH_REDUCTION = 0.65  # Reduce hatch linewidth by 35%

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_wind_data(filepath):
    """Load and prepare wind data for SSW detection."""
    ds = xr.open_dataset(filepath).squeeze()
    ds = ds.load()  # Load data into memory
    vortex = ds.sel(time=slice(TIME_START, TIME_END))
    ssw_subset = vortex.sel(time=vortex.time.dt.month.isin(SSW_MONTHS))
    return ds, vortex, ssw_subset


def identify_ssw_events(ssw_subset, vortex):
    """
    Identify Sudden Stratospheric Warming events from wind data.
    
    Returns list of SSW event dates.
    """
    years = set(ssw_subset.time.dt.year.values)
    indexdict = {year: [] for year in years}
    
    for year in years:
        tmp = ssw_subset.sel(time=slice(f'{int(year)}-11-01', f'{int(year+1)}-3-31'))
        dates = tmp.time.values
        reversals = [i for i, v in enumerate(tmp.uwnd.values) if v < 0]
        dayswitheasterlies = [list(group) for group in mit.consecutive_groups(reversals)]
        
        # Single group case
        if len(dayswitheasterlies) == 1:
            firstvalue = dayswitheasterlies[0][0]
            lastvalue = dayswitheasterlies[0][-1]
            lastvaluedate = dates[lastvalue]
            lastvaluedate = pd.date_range(lastvaluedate, f'{int(year+1)}-4-29')
            windbeforeapril30 = vortex.sel(time=lastvaluedate).uwnd.values
            westerlies = [i for i, v in enumerate(windbeforeapril30) if v > 0]
            if len(westerlies) > 0:
                westerlygroups = [list(group) for group in mit.consecutive_groups(westerlies)]
                westerlygroupslength = [len(group) for group in westerlygroups]
                maxlength = np.nanmax(westerlygroupslength)
                if maxlength > MIN_WESTERLY_DAYS - 1:
                    indexdict[year].append(dates[firstvalue])
        
        # Multiple groups case
        if len(dayswitheasterlies) > 1:
            firstvalue = dayswitheasterlies[0][0]
            lastvalue = dayswitheasterlies[0][-1]
            lastvaluedate = dates[lastvalue]
            daysbeforeapril30 = pd.date_range(lastvaluedate, f'{int(year+1)}-4-29')
            windbeforeapril30 = vortex.sel(time=daysbeforeapril30).uwnd.values
            westerlies = [i for i, v in enumerate(windbeforeapril30) if v > 0]
            if len(westerlies) > 0:
                westerlygroups = [list(group) for group in mit.consecutive_groups(westerlies)]
                westerlygroupslength = [len(group) for group in westerlygroups]
                maxlength = np.nanmax(westerlygroupslength)
                if maxlength > MIN_WESTERLY_DAYS - 1:
                    indexdict[year].append(dates[firstvalue])
            
            # Check for additional SSWs
            for i, v in enumerate(dayswitheasterlies):
                if i + 1 == len(dayswitheasterlies):
                    break
                
                currentgroup = dayswitheasterlies[int(i)]
                first_currentgroup = currentgroup[0]
                last_currentgroup = currentgroup[-1]
                
                nextgroup = dayswitheasterlies[int(i+1)]
                first_nextgroup = nextgroup[0]
                
                if first_nextgroup - last_currentgroup > MIN_GROUP_SEPARATION:
                    date_first_nextgroup = dates[first_nextgroup]
                    daysbeforeapril30 = pd.date_range(date_first_nextgroup, f'{int(year+1)}-4-29')
                    windbeforeapril30 = vortex.sel(time=daysbeforeapril30).uwnd.values
                    westerlies = [i for i, v in enumerate(windbeforeapril30) if v > 0]
                    if len(westerlies) > 0:
                        westerlygroups = [list(group) for group in mit.consecutive_groups(westerlies)]
                        westerlygroupslength = [len(group) for group in westerlygroups]
                        maxlength = np.nanmax(westerlygroupslength)
                        if maxlength > MIN_WESTERLY_DAYS - 1:
                            indexdict[year].append(dates[first_nextgroup])
    
    # Collect all events
    events = []
    for year in years:
        for v in indexdict[year]:
            events.append(v)
    
    return events


def prepare_data(filepath, var_name, unit_conversion):
    """
    Load and prepare data (SLP or temperature).
    
    Returns climatology, transient anomalies, and climatological wave.
    """
    zg = xr.open_dataset(filepath)
    zg = zg.load()  # Load data into memory
    zg = zg / unit_conversion  # Apply unit conversion
    
    # Daily seasonal cycle
    climatology = zg.groupby("time.dayofyear").mean("time")
    
    # Transient anomalies (deviation from daily climatology)
    transientwave = zg.groupby("time.dayofyear") - climatology
    
    # Climatological wave (zonally asymmetric component)
    climatologicalwave = (zg - zg.mean("lon")).groupby("time.dayofyear").mean("time")
    
    # Create storage array for climatological wave
    storage = xr.zeros_like(transientwave)
    storagedata = storage[var_name].values
    storagedates = pd.date_range(storage.time.values[0], storage.time.values[-1])
    
    for i, v in enumerate(storagedates):
        tmp = v.dayofyear
        storagedata[i, :, :] = climatologicalwave.sel(dayofyear=tmp)[var_name].values
    
    storage[var_name].values = storagedata
    
    return zg, transientwave, storage


def extract_ssw_windows(events, transientwave, storage, var_name):
    """
    Extract 60-day windows around each SSW event.
    
    Returns arrays of climatological wave and transient anomalies for each event.
    """
    lag_low = []
    lag_high = []
    
    for val in events:
        tmp = pd.Timestamp(str(val))
        first = tmp - timedelta(days=DAYS_BEFORE_SSW)
        second = first + timedelta(days=TOTAL_DAYS)
        lag_low.append(first)
        lag_high.append(second)
    
    sswclimwave = []
    sswtransient = []
    
    for i, v in enumerate(lag_low):
        tmp = storage.sel(time=slice(lag_low[i], lag_high[i]))[var_name].values
        sswclimwave.append(tmp)
        
        tmp = transientwave.sel(time=slice(lag_low[i], lag_high[i]))[var_name].values
        sswtransient.append(tmp)
    
    sswtransient = np.array(sswtransient)
    sswclimwave = np.array(sswclimwave)
    
    return sswtransient, sswclimwave


def calculate_period_composites(sswtransient, sswclimwave, periods):
    """
    Calculate 15-day averaged composites for each period.
    
    Returns arrays of transient and climatological composites.
    """
    store = []
    clim = []
    
    for start_idx, end_idx in periods:
        # Average over 15-day period
        tmp = np.nanmean(sswclimwave[:, start_idx:end_idx], axis=1)
        tmp = np.nanmean(tmp, axis=0)
        clim.append(tmp)
        
        tmp = np.nanmean(sswtransient[:, start_idx:end_idx], axis=1)
        tmp = np.nanmean(tmp, axis=0)
        store.append(tmp)
    
    store = np.array(store)
    clim = np.array(clim)
    
    return store, clim


def calculate_significance(sswtransient, transientwave, events, periods, var_name):
    """
    Calculate statistical significance by comparing SSW composites to same calendar dates across all years.
    
    Returns significance masks for each period.

    Are SSW patterns significantly different from the same calendar dates in other years?" 
    This accounts for natural interannual variability at that time of year.

    The logic:
    1. For each SSW event, extract its 60-day window (29 days before SSW)
    2. For each SSW event, extract the SAME calendar dates from all other years
    3. Build a null distribution from all these "same calendar date" samples
    4. Compare the SSW composite to this null distribution
    5. If SSW composite falls outside the 95% confidence interval → significant
    
    """
    all_years = np.unique([pd.Timestamp(event).year for event in events])
    
    significance_masks = []
    
    for period_idx, (start_idx, end_idx) in enumerate(periods):
        period_label = PERIOD_LABELS[period_idx] if period_idx < len(PERIOD_LABELS) else f"Period {period_idx+1}"
        print(f"    Processing {period_label}...", end=' ', flush=True)
        
        # Get SSW composite
        ssw_period_data = np.nanmean(sswtransient[:, start_idx:end_idx, :, :], axis=1)
        observed_composite = np.nanmean(ssw_period_data, axis=0)
        
        # Build null distribution
        null_composites = []
        
        for event_idx, event_date in enumerate(events):
            event_timestamp = pd.Timestamp(event_date)
            event_start = event_timestamp - timedelta(days=DAYS_BEFORE_SSW)
            null_data_list = []
            
            for year in all_years:
                if year == event_timestamp.year:
                    continue
                
                try:
                    year_start = pd.Timestamp(year=year, month=event_start.month, day=event_start.day)
                    year_end = year_start + timedelta(days=TOTAL_DAYS)
                    
                    data_start = pd.Timestamp(transientwave.time.values[0])
                    data_end = pd.Timestamp(transientwave.time.values[-1])
                    
                    if (year_start >= data_start and year_end <= data_end):
                        year_data = transientwave.sel(time=slice(year_start, year_end))
                        year_period_data = np.nanmean(year_data[var_name].values[start_idx:end_idx, :, :], axis=0)
                        null_data_list.append(year_period_data)
                        
                except (ValueError, KeyError):
                    continue
            
            if len(null_data_list) > 0:
                null_composites.extend(null_data_list)
        
        null_composites = np.array(null_composites)
        
        # Bootstrap null distribution
        n_null = null_composites.shape[0]
        bootstrap_means = []
        
        for boot_iter in range(N_BOOTSTRAP):
            bootstrap_indices = np.random.choice(n_null, size=n_null, replace=True)
            bootstrap_sample = null_composites[bootstrap_indices]
            bootstrap_mean = np.nanmean(bootstrap_sample, axis=0)
            bootstrap_means.append(bootstrap_mean)
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence intervals
        lower_bound = np.percentile(bootstrap_means, (ALPHA/2)*100, axis=0)
        upper_bound = np.percentile(bootstrap_means, (1-ALPHA/2)*100, axis=0)
        
        # Test significance
        significant = (observed_composite > upper_bound) | (observed_composite < lower_bound)
        significance_masks.append(significant)
        
        n_sig = np.sum(significant)
        total = significant.size
        print(f"done ({n_sig}/{total} grid points significant)")
    
    significance_masks = np.array(significance_masks)
    
    return significance_masks


def create_plot(store, significance_masks, zg, events, config):
    """Create and save the composite plot."""
    # Set up figure
    fig = plt.figure(figsize=FIG_SIZE)
    mpl.rcParams['font.sans-serif'].insert(0, FONT_NAME)
    plt.suptitle(f'{config["var_label"]} anomalies ({config["unit_label"]}) during \n major Sudden Stratospheric Warmings ({len(events)} SSWs)', 
                 fontsize=SUPTITLE_FONTSIZE, y=0.98)
    
    # Colorbar settings
    vlevs = np.linspace(config['vmin'], config['vmax'], num=config['n_levels'])
    if 0.0 not in vlevs:
        vlevs = np.sort(np.append(vlevs, 0.0))
    ticks = [config['vmin'], config['vmin']/2, 0, config['vmax']/2, config['vmax']]
    
    # Colormap
    cmap = config['colormap']
    norm = mcolors.BoundaryNorm(boundaries=vlevs, ncolors=cmap.N, clip=True)
    
    # Subplot layout
    nrows, ncols = 2, 2
    
    # Create subplots
    for idx in range(len(PERIOD_LABELS)):
        ax = plt.subplot(nrows, ncols, idx + 1, 
                         projection=ccrs.Orthographic(central_longitude=0, central_latitude=90))
        
        ax.set_title(f'{PERIOD_LABELS[idx]}', fontsize=TITLE_FONTSIZE, pad=3)
        
        # Add map features
        ax.coastlines(linewidth=COASTLINE_LINEWIDTH, resolution=COASTLINE_RESOLUTION, 
                     color=COASTLINE_COLOR, alpha=COASTLINE_ALPHA)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=LAND_ALPHA, zorder=0)
        ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
        
        # Circular boundary
        theta = np.linspace(0, 2*np.pi, CIRCLE_POINTS)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)
        
        # Prepare data
        data = store[idx]
        if data.ndim > 2:
            data = np.squeeze(data)
        
        cyclic_data, cyclic_lon = add_cyclic_point(data, coord=zg.lon.values)
        if cyclic_data.ndim > 2:
            cyclic_data = np.squeeze(cyclic_data)
        
        lat_vals = zg.lat.values
        lon_2d, lat_2d = np.meshgrid(cyclic_lon, lat_vals)
        
        # Plot contours
        im = ax.contourf(lon_2d, lat_2d, cyclic_data,
                        levels=vlevs,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, extend='both',
                        zorder=1,
                        antialiased=True)
        
        # Add stippling for non-significant regions
        if idx < len(significance_masks):
            sig_mask = significance_masks[idx]
            if sig_mask.ndim > 2:
                sig_mask = np.squeeze(sig_mask)
            
            cyclic_sig, _ = add_cyclic_point(sig_mask, coord=zg.lon.values)
            if cyclic_sig.ndim > 2:
                cyclic_sig = np.squeeze(cyclic_sig)
            
            sig_lon_2d, sig_lat_2d = np.meshgrid(cyclic_lon, lat_vals)
            sig_plot = np.where(cyclic_sig, 0, 1)
            if sig_plot.ndim > 2:
                sig_plot = np.squeeze(sig_plot)
            
            original_hatch_lw = mpl.rcParams.get('hatch.linewidth', 1.0)
            mpl.rcParams['hatch.linewidth'] = original_hatch_lw * HATCH_LINEWIDTH_REDUCTION
            
            ax.contourf(sig_lon_2d, sig_lat_2d, sig_plot,
                       levels=[0.5, 1.5], 
                       colors='none',
                       hatches=['...'],
                       transform=ccrs.PlateCarree(),
                       zorder=10,
                       alpha=0.0,
                       linewidths=0)
        
        ax.set_xticks([], [])
        ax.set_yticks([], [])
    
    # Add colorbar
    cb_ax = fig.add_axes([0.35, 0.02, 0.30, 0.015])
    cbar = fig.colorbar(im, cax=cb_ax, ticks=ticks, orientation='horizontal')
    cbar.ax.tick_params(labelsize=9, width=0.5, length=3)
    cbar.ax.set_xticklabels(ticks, weight='normal')
    cbar.set_label(f'{config["var_label"].title()} Anomaly ({config["unit_label"]})', 
                   fontsize=CBAR_FONTSIZE, labelpad=5)
    
    # Adjust layout
    plt.subplots_adjust(top=0.88, bottom=0.04, hspace=0.08, wspace=0.0, 
                        left=0.02, right=0.98)
    
    # Save figure
    plt.savefig(config['output_file'], dpi=DPI, bbox_inches='tight', 
                facecolor='white', transparent=False)


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run SSW wave analysis."""
    print(f"Analyzing {CONFIG['var_label'].lower()} anomalies during SSWs...")
    print("=" * 60)
    
    # Step 1: Identify SSW events
    print("Step 1: Identifying SSW events from wind data...")
    ds, vortex, ssw_subset = load_wind_data(U10_FILE)
    events = identify_ssw_events(ssw_subset, vortex)
    print(f"  Found {len(events)} SSW events")
    print("  SSW central dates:")
    for i, event in enumerate(events, 1):
        event_date = pd.Timestamp(event).strftime('%Y-%m-%d')
        print(f"    {i:2d}. {event_date}")
    
    # Step 2: Prepare data
    print(f"\nStep 2: Loading and preparing {CONFIG['var_label'].lower()} data...")
    zg, transientwave, storage = prepare_data(
        CONFIG['file_path'], 
        CONFIG['var_name'], 
        CONFIG['unit_conversion']
    )
    print(f"  Data loaded: {len(transientwave.time)} time steps")
    
    # Step 3: Extract SSW windows
    print(f"\nStep 3: Extracting 60-day windows around {len(events)} SSW events...")
    sswtransient, sswclimwave = extract_ssw_windows(
        events, transientwave, storage, CONFIG['var_name']
    )
    print(f"  Extracted windows: {sswtransient.shape[0]} events × {sswtransient.shape[1]} days")
    
    # Step 4: Calculate period composites
    print(f"\nStep 4: Calculating 15-day averaged composites for {len(PERIODS)} periods...")
    store, clim = calculate_period_composites(sswtransient, sswclimwave, PERIODS)
    print("  Composites calculated")
    
    # Step 5: Calculate statistical significance
    print(f"\nStep 5: Calculating statistical significance (bootstrap: {N_BOOTSTRAP} iterations)...")
    significance_masks = calculate_significance(
        sswtransient, transientwave, events, PERIODS, CONFIG['var_name']
    )
    print("  Significance testing complete")
    
    # Step 6: Create plot
    print(f"\nStep 6: Creating plot and saving to {CONFIG['output_file']}...")
    create_plot(store, significance_masks, zg, events, CONFIG)
    print("  Plot saved successfully")
    print("=" * 60)
    print("Analysis complete!")


# ============================================================================
# MAIN BLOCK
# ============================================================================

if __name__ == '__main__':
    main()

