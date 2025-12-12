# SSW Surface Anomalies Analysis

Analyzes surface anomalies during Sudden Stratospheric Warmings (SSWs) using 15-day averaging periods with statistical significance testing. Can seamlessly switch between sea level pressure and surface air temperature analysis.

## Features

- Identifies SSW events from zonal wind reversals at 60°N
- Extracts 60-day windows (29 days before to 30 days after SSW)
- Calculates 15-day averaged composites for four periods: days 29-15 before SSW, days 14-0 before SSW, days 1-15 after SSW, and days 16-30 after SSW
- Performs statistical significance testing by comparing SSW composites to same calendar dates in non-SSW years
- Creates publication-quality maps with stippling for non-significant regions

## Usage

Set `VARIABLE_TYPE = 'slp'` or `'temp'` at the top of the script to switch between sea level pressure and surface temperature analysis. Configure file paths in the constants section, then run:

```bash
python ssw_surface_anomalies_15day.py
```

## Statistical Method

Significance is determined by comparing each SSW event's 60-day period to the same calendar dates across all other years, accounting for natural interannual variability. Bootstrap resampling (1000 iterations) establishes 95% confidence intervals. Non-statistically significant regions are stippled on the maps.

