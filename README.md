# climate-analytics-portfolio
Selected examples of my scientific computing, data engineering, and climate analytics code.

QBO metrics (signal extraction, reproducible analysis): Computes standardized Quasi-Biennial Oscillation (QBO) diagnostics using modular xarray-based functions and provides reproducible workflow for downstream analysis. See the "qbo-hist-amip-summary-v4.png" figure for mult-model example.

SSW composites (pipeline + analysis + visualization): An end-to-end pipeline that constructs composite fields of surface temperature or sea level pressure anomalies following SSW events using xarray/dask, including diagnostics and plotting utilities.

NOAA GEFS forecast data multivariate downloader (data engineering + workflow automation): A production-style Python script that automatically retrieves all GEFS ensemble members for five variables for a given forecast cycle using parallelization and proactive memory management, handles file management, retries, and structured output.

Link to download sample NCEP1 reanalysis data, data which is used in "qbo-metrics" and "ssw-composites": https://drive.google.com/drive/folders/13DhK1sMYJRz9jQKAZQK0BMLwukhCBdYp?usp=sharing
