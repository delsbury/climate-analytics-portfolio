# GEFS Forecast Data Downloader

Downloads and processes NOAA GEFS (Global Ensemble Forecast System) forecast data, extracting multiple variables for all 31 ensemble members and concatenating them into netCDF files.

## Features

- Downloads 5 variables: 2m temperature, 10m u-wind, mean sea level pressure, total precipitation, and downward shortwave radiation
- Processes forecasts from 6 to 240 hours (every 6 hours)
- Handles all 31 ensemble members (control + 30 perturbed members)
- Optimized for memory efficiency with sequential variable processing
- Uses parallel downloads for faster data retrieval
- Saves concatenated data to netCDF4 format with compression

## Usage

Configure the date, cycle, and output directory in the `main()` function, then run:

```bash
python save_gefs_netcdf_forecast.py
```

## Output

Creates one netCDF file per variable containing all ensemble members and forecast hours concatenated along member and forecast_hour dimensions. Files are saved with naming convention: `gefs_{variable}_all_members_{date}_{cycle}z_f{min}_to_f{max}_v3.nc`

## Requirements

- herbie (for GEFS data access)
- xarray
- netCDF4
- numpy

