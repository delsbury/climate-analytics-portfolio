# climate-analytics-portfolio
Welcome! My name is Dillon Elsbury - I am a geospatial data scientist with expertise in atmospheric science and climate. This repo contains my CV (Elsbury_CV_Dec_2025.pdf) and selected examples of my climate analytics, scientific computing, and data engineering workflows. 

-- QBO metrics (signal extraction, reproducible analysis): Computes standardized Quasi-Biennial Oscillation (QBO) diagnostics using modular xarray-based functions and provides reproducible workflow for downstream analysis. See the "qbo-hist-amip-summary-v4.png" figure for multi-model example.

-- SSW composites (pipeline + analysis + visualization): An end-to-end pipeline that constructs composite fields of surface temperature or sea level pressure anomalies following major Sudden Stratospheric Warming (SSW) events using xarray/dask, including diagnostics and plotting utilities.

-- NOAA GEFS forecast data multivariate downloader (data engineering + workflow automation): A production-style Python script that automatically retrieves all NOAA Global Ensemble Forecast System (GEFS) ensemble members for five variables for a given forecast cycle using parallelization and proactive memory management, handles file management, retries, and structured output.

Link to download sample NCEP1 reanalysis data, data which is used in "qbo-metrics" and "ssw-composites": https://drive.google.com/drive/folders/13DhK1sMYJRz9jQKAZQK0BMLwukhCBdYp?usp=sharing

I have also included a recent unpublished manuscript: Elsbury_2026_process_oriented_diagnostics.pdf. This paper provides a comprehensive evaluation of the large-scale atmospheric circulation in 40+ CMIP6 and AMIP6 models, showing how differences in ocean boundary conditions and the resulting sea-surface temperature biases, drive systematic errors in jets, polar vortices, wave dynamics, and ENSO/QBO teleconnections relative to reanalysis. This project required managing a ~100 TB Netcdf/Zarr data lake.
