"""
QBO Metrics Calculator - Professional Version

Calculates Quasi-Biennial Oscillation (QBO) metrics for zonal wind datasets.
Based on methods from Pascoe et al. (2005) and Richter et al. (2020).

This version is optimized with:
- Constants at the top for easy configuration
- Modular functions for better maintainability
- Main block for script execution
"""

import numpy as np
import xarray as xr
from scipy.fft import fft, fftfreq
from scipy import interpolate
from scipy.optimize import curve_fit

# ============================================================================
# CONSTANTS
# ============================================================================

# QBO cycle constraints
MIN_CYCLE_LENGTH_MONTHS = 14  # Minimum valid QBO cycle length (months)

# Geographic bounds for tropical averaging
TROPICAL_LAT_MIN = -5.0  # degrees
TROPICAL_LAT_MAX = 5.0   # degrees

# Smoothing window
SMOOTHING_WINDOW_MONTHS = 5  # Running mean window (months)

# Interpolation parameters
INTERP_POINTS = 10000  # Number of points for high-resolution interpolation

# Vertical extent calculation
SURFACE_PRESSURE_HPA = 1000.0  # hPa
ALTITUDE_SCALE_FACTOR = -7000.0  # meters

# Fourier amplitude thresholds
FOURIER_HALF_MAX_RATIO = 0.5  # 50% of max for vertical extent
FOURIER_BASE_RATIO = 0.1  # 10% of max for lowest level

# Date range for data filtering
START_DATE = '1979-01-01'  # Start date for time slice (format: 'YYYY-MM-DD')
END_DATE = '2009-12-01'    # End date for time slice (format: 'YYYY-MM-DD')

# QBO pressure level
QBO_ISOBAR_PRESSURE_HPA = 20.0  # Pressure level (hPa) at which to calculate QBO metrics

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_dataset(ds):
    """
    Ensure dataset has correct coordinate names and pressure level ordering.
    
    Automatically renames common coordinate/variable names to expected format:
    - 'level' or 'plev' -> 'lev'
    - 'latitude' -> 'lat'
    - 'longitude' -> 'lon' (optional - may be removed after averaging)
    - 'uwnd' or 'u' -> 'ua'
    
    Required dimensions: 'time', 'lev', 'lat'
    Optional dimensions: 'lon' (may be absent if already averaged over longitude)
    
    Also ensures pressure levels are in descending order (high to low).
    """
    # Check and rename coordinates if needed
    rename_dict = {}
    
    # Check level/plev coordinate
    if 'lev' not in ds.dims and 'lev' not in ds.coords:
        if 'level' in ds.dims or 'level' in ds.coords:
            rename_dict['level'] = 'lev'
        elif 'plev' in ds.dims or 'plev' in ds.coords:
            rename_dict['plev'] = 'lev'
    
    # Check latitude coordinate
    if 'lat' not in ds.dims and 'lat' not in ds.coords:
        if 'latitude' in ds.dims or 'latitude' in ds.coords:
            rename_dict['latitude'] = 'lat'
    
    # Check longitude coordinate
    if 'lon' not in ds.dims and 'lon' not in ds.coords:
        if 'longitude' in ds.dims or 'longitude' in ds.coords:
            rename_dict['longitude'] = 'lon'
    
    # Check and rename data variable if needed
    if 'ua' not in ds.data_vars and 'ua' not in ds.variables:
        if 'uwnd' in ds.data_vars or 'uwnd' in ds.variables:
            rename_dict['uwnd'] = 'ua'
        elif 'u' in ds.data_vars or 'u' in ds.variables:
            rename_dict['u'] = 'ua'
    
    # Apply renames if any were needed
    if rename_dict:
        ds = ds.rename(rename_dict)
        print(f"Renamed coordinates/variables: {rename_dict}")
    
    # Validate required coordinates exist
    # Note: 'lon' is optional as dataset may have already been averaged over longitude
    required_dims = ['time', 'lev', 'lat']
    optional_dims = ['lon']  # Optional - may be removed after averaging
    missing_dims = [dim for dim in required_dims if dim not in ds.dims]
    if missing_dims:
        raise ValueError(f"Missing required dimensions: {missing_dims}. "
                        f"Available dimensions: {list(ds.dims)}")
    
    # Validate required variable exists
    if 'ua' not in ds.data_vars and 'ua' not in ds.variables:
        raise ValueError(f"Missing required variable 'ua'. "
                        f"Available variables: {list(ds.data_vars)}")
    
    # Ensure pressure levels are in descending order (high to low)
    if ds.lev.values[-1] > ds.lev.values[0]:
        ds = ds.reindex(lev=list(reversed(ds.lev)))
    
    return ds


def extract_tropical_qbo(ds, qbo_pressure_level):
    """
    Extract QBO signal at specified pressure level, averaged over tropical region.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with dimensions (time, lev, lat, lon)
    qbo_pressure_level : float
        Pressure level in hPa to extract QBO signal
        
    Returns
    -------
    qbo : xarray.DataArray
        Time series of QBO signal at specified level, latitudinally weighted
    """
    # Interpolate to specified pressure level
    subset = ds.interp(lev=qbo_pressure_level)
    
    # Select tropical region (5S-5N)
    tropical_mask = np.logical_and(
        subset.lat >= TROPICAL_LAT_MIN,
        subset.lat <= TROPICAL_LAT_MAX
    )
    tropical_subset = subset.isel(lat=tropical_mask)
    
    # Calculate latitudinally weighted average
    qbo = tropical_subset.mean('lat')
    weights = np.cos(np.deg2rad(tropical_subset.lat.values))
    weighted_sum = np.sum(
        tropical_subset.ua.values * weights[np.newaxis, :],
        axis=1
    )
    qbo_values = weighted_sum / np.sum(weights)
    qbo.ua.values[:] = qbo_values[:]
    qbo = qbo.ua
    
    print('qbo')
    
    # Apply smoothing
    qbo = qbo.rolling(
        time=SMOOTHING_WINDOW_MONTHS,
        center=True
    ).mean()
    
    return qbo


def identify_qbo_cycles(qbo_time_series):
    """
    Identify complete QBO cycles from zero crossings.
    
    Parameters
    ----------
    qbo_time_series : xarray.DataArray
        Smoothed QBO time series
        
    Returns
    -------
    cycles : list
        List of arrays, each representing a complete QBO cycle
    periods : list
        List of cycle periods in months
    """
    # Identify zero crossings (phase changes)
    zero_crossings = np.where(np.diff(np.sign(qbo_time_series.values)))[0]
    
    store = []
    cycles = []
    periods = []
    
    for i, v in enumerate(zero_crossings):
        # Append pairs of easterly/westerly winds to store
        if i != 0:
            segment = qbo_time_series.values[
                zero_crossings[i - 1] + 1:zero_crossings[i] + 1
            ]
            store.append(segment)
        
        # Every even index represents a complete cycle
        if i != 0 and i % 2 == 0:
            complete_cycle = np.concatenate((store[-2], store[-1]))
            
            # Require minimum cycle length
            if len(complete_cycle) >= MIN_CYCLE_LENGTH_MONTHS:
                cycles.append(complete_cycle)
                periods.append(len(complete_cycle))
    
    return cycles, periods


def calculate_period_statistics(periods):
    """Calculate minimum, mean, and maximum QBO cycle periods."""
    period_min = np.round(np.nanmin(periods), 1)
    period_max = np.round(np.nanmax(periods), 1)
    period_mean = np.round(np.nanmean(periods), 1)
    return period_min, period_mean, period_max


def calculate_amplitude_statistics(cycles):
    """
    Calculate easterly, westerly, and total QBO amplitudes.
    
    Following Richter et al. (2020) definition.
    """
    easterly_amp = np.round(np.nanmean([np.nanmin(v) for v in cycles]), 1)
    westerly_amp = np.round(np.nanmean([np.nanmax(v) for v in cycles]), 1)
    
    # Total QBO amplitude
    qbo_amp = np.abs(easterly_amp / 2) + np.abs(westerly_amp / 2)
    qbo_amp = np.round(qbo_amp, 1)
    
    return easterly_amp, westerly_amp, qbo_amp


def calculate_fourier_amplitude(ds, uwnd, period_min, period_max):
    """
    Calculate QBO Fourier amplitude following Pascoe et al. (2005).
    
    Returns the ratio of QBO power spectrum to full zonal wind power spectrum,
    multiplied by standard deviation of zonal wind.
    """
    # Standard deviation across entire dataset (over time axis)
    # Environment-proof calculation: explicitly compute std over time dimension
    # Some numpy versions may have issues with axis parameter, so compute explicitly
    # uwnd shape is (time, lev, lat) after mean('lon')
    
    # Method 1: Try standard np.nanstd first
    std = np.nanstd(uwnd, axis=0)
    
    # Validate std is not all zeros (environment-proof check)
    if np.allclose(std, 0.0) or np.all(np.isnan(std)):
        # Method 2: Compute manually using explicit loop over spatial dimensions
        # This ensures we compute std over time for each (lev, lat) point
        if uwnd.ndim == 3:
            time_dim, lev_dim, lat_dim = uwnd.shape
            std = np.zeros((lev_dim, lat_dim), dtype=uwnd.dtype)
            for lev_idx in range(lev_dim):
                for lat_idx in range(lat_dim):
                    time_series = uwnd[:, lev_idx, lat_idx]
                    std[lev_idx, lat_idx] = np.nanstd(time_series)
        elif uwnd.ndim == 4:
            time_dim, lev_dim, lat_dim, lon_dim = uwnd.shape
            std = np.zeros((lev_dim, lat_dim, lon_dim), dtype=uwnd.dtype)
            for lev_idx in range(lev_dim):
                for lat_idx in range(lat_dim):
                    for lon_idx in range(lon_dim):
                        time_series = uwnd[:, lev_idx, lat_idx, lon_idx]
                        std[lev_idx, lat_idx, lon_idx] = np.nanstd(time_series)
        else:
            # Fallback: compute mean and std manually
            mean_vals = np.nanmean(uwnd, axis=0)
            # Reshape for broadcasting: mean_vals needs to match uwnd shape
            mean_broadcast = np.broadcast_to(mean_vals, uwnd.shape)
            squared_deviations = (uwnd - mean_broadcast) ** 2
            std = np.sqrt(np.nanmean(squared_deviations, axis=0))
    
    # Filter frequencies corresponding to QBO periods (match original code)
    freq = 1 / fftfreq(len(uwnd))
    qbo_freq_mask = (freq > period_min) & (freq < period_max)
    arr = np.where(qbo_freq_mask)[0]
    
    # FFT and power spectrum
    fft_result = fft(uwnd, axis=0)
    amplitudes = np.power(np.abs(fft_result)[:len(fft_result) // 2], 2)
    n_amplitudes = len(amplitudes)
    
    # Filter arr to only include valid indices for amplitudes array
    arr = arr[arr < n_amplitudes]
    
    # Calculate power spectrum ratio for each level (matching original exactly)
    quotients = []
    for i, v in enumerate(ds.lev.values):
        qbodata = np.nansum(amplitudes[arr, i], axis=0)
        alldata = np.nansum(amplitudes[1:, i], axis=0)
        quot = np.true_divide(qbodata, alldata)
        quotients.append(quot)
    
    filtered = np.array(quotients)
    
    # Fourier amplitude = ratio * standard deviation
    fa = np.multiply(filtered, std)
    
    return fa


def calculate_vertical_extent(ds, height_profile, hmax_array):
    """
    Calculate vertical extent and lowest QBO level.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    height_profile : ndarray
        Latitudinally averaged Fourier amplitude profile
    hmax_array : ndarray
        Array of reference pressure level index (from np.where)
        
    Returns
    -------
    vertical_extent : float
        Vertical extent in kilometers
    lowest_lev : float
        Lowest pressure level (hPa) where QBO extends
    """
    # Calculate thresholds
    half_max = np.max(height_profile) / 2
    qbo_base = np.max(height_profile) * 0.1
    
    # Interpolate for higher resolution (matching original exactly)
    f = interpolate.interp1d(ds.lev.values, height_profile)
    xnew = np.linspace(ds.lev.values[0], ds.lev.values[-1], num=10000)
    ynew = f(xnew)
    
    # Find reference level in interpolated array (matching original: ds.lev.values[hmax])
    # Extract integer index explicitly for reliable behavior across numpy versions
    hmax_int = int(hmax_array[0]) if len(hmax_array) > 0 else int(hmax_array)
    hmax_lev_value = float(ds.lev.values[hmax_int])
    hmax_idx = (np.abs(xnew - hmax_lev_value)).argmin()
    
    # Split into lower and upper portions (matching original exactly)
    lower_portion = ynew[:hmax_idx]
    upper_portion = ynew[hmax_idx:]
    
    lower_portion_isobar = xnew[:hmax_idx]
    upper_portion_isobar = xnew[hmax_idx:]
    
    # Find half-maximum points (matching original)
    lower_vertical_extent = (np.abs(lower_portion - half_max)).argmin()
    upper_vertical_extent = (np.abs(upper_portion - half_max)).argmin()
    
    # Get pressure levels at half-maximum
    bottom = lower_portion_isobar[lower_vertical_extent]
    top = upper_portion_isobar[upper_vertical_extent]
    
    # Convert pressure to altitude (matching original exactly)
    sfc = 1000  # hPa
    bottomz = np.log(bottom / sfc) * -7000
    topz = np.log(top / sfc) * -7000
    
    # Calculate vertical extent in kilometers
    vertical_extent = (topz - bottomz) / 1000
    vertical_extent = np.round(vertical_extent, 1)
    
    # Find lowest level (10% of max) - matching original: lower_portion_isobar
    lowest_lev = lower_portion_isobar[(np.abs(lower_portion - qbo_base)).argmin()]
    lowest_lev = np.round(lowest_lev, 1)
    
    return vertical_extent, lowest_lev


def calculate_latitudinal_extent(ds, fa, hmax_array):
    """
    Calculate latitudinal extent using Gaussian fit.
    
    Returns the full width at half maximum of a Gaussian fit to the
    Fourier amplitude at the reference level.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    fa : ndarray
        Fourier amplitude array
    hmax_array : ndarray
        Array of reference pressure level index (from np.where)
    """
    # Match original code exactly: fa[hmax][0]
    # Note: hmax_array is from np.where, so it's an array like [16]
    # Extract integer index explicitly for reliable behavior across numpy versions
    hmax_int = int(hmax_array[0]) if len(hmax_array) > 0 else int(hmax_array)
    xdata = ds.lat.values
    ydata = fa[hmax_int]
    ydata[0] = 0
    ydata[-1] = 0

    # Recast xdata and ydata into numpy arrays so we can use their handy features
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    # Gaussian function
    def gauss(x, H, A, x0, sigma):
        return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))
    
    # Fit Gaussian (matching original code exactly)
    def gauss_fit(x, y):
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(
            gauss, x, y,
            p0=[min(y), max(y), mean, sigma]
        )
        return popt
    
    out = gauss(xdata, *gauss_fit(xdata, ydata))
    
    # Interpolate for higher resolution (matching original exactly)
    f = interpolate.interp1d(ds.lat.values, out)
    xnew = np.linspace(ds.lat.values[0], ds.lat.values[-1], num=INTERP_POINTS)
    ynew = f(xnew)
    
    # Split and find half-maximum latitudes (matching original: split at 5000)
    lower_portion = ynew[:5000]
    upper_portion = ynew[5000:]
    
    lower_portion_lat = xnew[:5000]
    upper_portion_lat = xnew[5000:]
    
    lat1 = lower_portion_lat[(np.abs(lower_portion - (np.max(out) / 2))).argmin()]
    lat2 = upper_portion_lat[(np.abs(upper_portion - (np.max(out) / 2))).argmin()]
    
    latitudinal_extent = np.abs(lat1) + np.abs(lat2)
    latitudinal_extent = np.round(latitudinal_extent, 1)
    
    return latitudinal_extent


def calculate_spatial_metrics(ds, qbo, fa):
    """
    Calculate all spatial metrics (vertical and latitudinal extent).
    
    Returns
    -------
    vertical_extent : float
        Vertical extent in kilometers
    lowest_lev : float
        Lowest pressure level (hPa)
    latitudinal_extent : float
        Latitudinal extent in degrees
    """
    # Find reference level index (matching original exactly)
    # Ensure qbo.lev.values is treated as scalar for comparison
    qbo_lev_value = float(qbo.lev.values) if hasattr(qbo.lev.values, '__float__') else qbo.lev.values
    hmax = np.where(ds.lev.values == qbo_lev_value)[0]
    print(hmax, 'hmax')
    
    # Retrieve the indices of lats between 5S and 5N (matching original exactly)
    lat_hits = [i for i, v in enumerate(ds.lat.values) if v >= -5 and v <= 5]
    
    # Retrieve the Fourier amplitude profile averaged latitudinally (w/weighting) between 5S and 5N
    # Matching original exactly with np.multiply, np.nansum, np.true_divide
    weights = np.cos(np.deg2rad(ds.lat.values[lat_hits]))
    interim = np.multiply(fa[:, lat_hits], weights[np.newaxis, :])
    interim2 = np.nansum(interim, axis=1)
    height_profile = np.true_divide(interim2, np.sum(weights))
    
    # Calculate vertical extent (hmax is already an array from np.where)
    vertical_extent, lowest_lev = calculate_vertical_extent(
        ds, height_profile, hmax
    )
    
    # Calculate latitudinal extent
    latitudinal_extent = calculate_latitudinal_extent(ds, fa, hmax)
    
    return vertical_extent, lowest_lev, latitudinal_extent


def format_output(period_min, period_mean, period_max, easterly_amp, westerly_amp,
                  qbo_amp, lowest_lev, vertical_extent, latitudinal_extent):
    """Format metrics into output lists."""
    metrics = [
        'minimum period: %s (months)' % period_min,
        'mean period: %s (months)' % period_mean,
        'maximum period: %s (months)' % period_max,
        'easterly amplitude: %s (m/s)' % easterly_amp,
        'westerly amplitude: %s (m/s)' % westerly_amp,
        'QBO amplitude: %s (m/s)' % qbo_amp,
        'lowest QBO level: %s (hPa)' % lowest_lev,
        'vertical extent: %s (kilometers)' % vertical_extent,
        'latitudinal extent of QBO: %s (degrees)' % latitudinal_extent
    ]
    
    metrics_raw = [
        period_min, period_mean, period_max, easterly_amp, westerly_amp,
        qbo_amp, lowest_lev, vertical_extent, latitudinal_extent
    ]
    
    return metrics, metrics_raw


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def qbo_metrics4(ds, QBOisobar):
    r"""Calculate Quasi-Biennial Oscillation (QBO) metrics for an input
    zonal wind dataset (dimensions = time x level x latitude x longitude)

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        The input DataArray or Dataset for which to calculate QBO diagnostics.
        Must have dimensions named 'time', 'lev', 'lat', 'lon' and variable 'ua'.
        If your data differs in dimension name (e.g., 'latitude'), use the rename method:
        ds.rename({'latitude':'lat'})
        ds.rename({'level':'lev'})
    QBOisobar : float
        Pressure level (hPa) at which to calculate QBO metrics.
        Default is QBO_ISOBAR_PRESSURE_HPA constant (typically 10 hPa).

    Returns
    -------
    metrics : list
        List of formatted metric strings describing each QBO metric
    metrics_raw : list
        List of raw metric values [period_min, period_mean, period_max,
        easterly_amp, westerly_amp, qbo_amp, lowest_lev, vertical_extent,
        latitudinal_extent]

    Notes
    -----
    period_min is required to be >= 14 months. This requirement is used because
    period_min and period_max are the time bounds that determine which frequencies
    are most "QBO-like." If period_min < 14 months (e.g., 12 months), the annual
    cycle zonal wind variability would be incorrectly deemed "QBO-like" and
    incorporated into the QBO Fourier amplitude calculations, rendering the
    Fourier amplitude and all QBO spatial metrics invalid.
    
    References
    ----------
    Pascoe et al. (2005) for Fourier amplitude calculations
    Richter et al. (2020, JGR) for amplitude definitions
    """
    print("Running the QBO metrics function 'qbo_metrics'")
    
    # Prepare dataset
    ds = prepare_dataset(ds)
    uwnd = ds.ua.values
    
    # Extract tropical QBO signal
    qbo = extract_tropical_qbo(ds, QBOisobar)
    
    # Identify QBO cycles
    cycles, periods = identify_qbo_cycles(qbo)
    
    # Calculate period statistics
    period_min, period_mean, period_max = calculate_period_statistics(periods)
    
    # Calculate amplitude statistics
    easterly_amp, westerly_amp, qbo_amp = calculate_amplitude_statistics(cycles)
    
    # Calculate Fourier amplitude
    fa = calculate_fourier_amplitude(ds, uwnd, period_min, period_max)
    
    # Calculate spatial metrics
    vertical_extent, lowest_lev, latitudinal_extent = calculate_spatial_metrics(
        ds, qbo, fa
    )
    
    # Determine QBO presence
    if period_min != period_max:
        print('Based on period statistics, dataset is likely to have a QBO')
        qbo_switch = 1
    else:
        print('Persistent stratospheric easterlies detected - dataset likely does not have QBO')
        qbo_switch = 0
    
    # Format and return output
    metrics, metrics_raw = format_output(
        period_min, period_mean, period_max, easterly_amp, westerly_amp,
        qbo_amp, lowest_lev, vertical_extent, latitudinal_extent
    )
    
    return metrics, metrics_raw


# ============================================================================
# MAIN BLOCK
# ============================================================================

if __name__ == '__main__':
    # Load dataset
    era5 = xr.open_dataset('uwnd.mon.mean.nc')
    
    # Prepare dataset: check and rename coordinates/variables to expected names
    era5 = prepare_dataset(era5)
    
    # Ensure latitude is in correct order (south to north)
    if era5.lat[0] > 0:
        era5 = era5.reindex(lat=era5.lat[::-1])
    
    # select time period using constants
    era5 = era5.sel(time=slice(START_DATE, END_DATE))
    
    print(era5)
    
    # Calculate QBO metrics using constant for pressure level
    # Average over longitude before passing to function
    qbo_status, metricsoutraw = qbo_metrics4(era5.mean('lon'), QBO_ISOBAR_PRESSURE_HPA)
    
    # Print results in table format
    print("\n" + "="*70)
    print("QBO METRICS RESULTS".center(70))
    print("="*70)
    
    # Define proper label formatting
    label_formatting = {
        'minimum period': 'Minimum Period',
        'mean period': 'Mean Period',
        'maximum period': 'Maximum Period',
        'easterly amplitude': 'Easterly Amplitude',
        'westerly amplitude': 'Westerly Amplitude',
        'qbo amplitude': 'QBO Amplitude',
        'lowest qbo level': 'Lowest QBO Level',
        'vertical extent': 'Vertical Extent',
        'latitudinal extent of qbo': 'Latitudinal Extent of QBO'
    }
    
    for metric in qbo_status:
        # Extract metric name and value from string like "minimum period: 22 (months)"
        parts = metric.split(': ')
        if len(parts) == 2:
            metric_key = parts[0].lower()
            metric_name = label_formatting.get(metric_key, parts[0].title())
            value_unit = parts[1]
            # Format as table row with proper alignment
            print(f"  {metric_name:<42} {value_unit:>25}")
    
    print("="*70)
    print(f"\nRaw values (for programmatic use): {metricsoutraw}")
