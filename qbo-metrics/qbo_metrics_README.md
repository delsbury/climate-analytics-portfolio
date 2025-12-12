# QBO Metrics Calculator

Calculates comprehensive Quasi-Biennial Oscillation (QBO) metrics from zonal wind datasets using methods from Schenzinger et al. (2017) and Pascoe et al. (2005) and Richter et al. (2020).

## Features

- Extracts QBO signal at specified pressure level (default 20 hPa)
- Identifies complete QBO cycles from zero crossings of monthly time-series of 20 hPa tropical (5S-5N) zonal wind
- Calculates period statistics (min, mean, max cycle lengths)
- Computes amplitude statistics (easterly, westerly, total QBO amplitude)
- Calculates Fourier amplitude using power spectrum analysis
- Determines spatial metrics: vertical extent, lowest QBO level, and latitudinal extent

## Usage

```python
from qbo_metrics import qbo_metrics4

metrics, metrics_raw = qbo_metrics4(dataset, QBOisobar=20.0)
```

## Input Requirements

Dataset must have dimensions: `time`, `lev`, `lat`, `lon` and variable `ua` (zonal wind). The script automatically handles common coordinate name variations.

## Output

Returns formatted metric strings and raw values including: cycle periods, amplitudes, vertical extent (km), lowest QBO level (hPa), and latitudinal extent (degrees).

## Configuration

Constants at the top allow easy adjustment of: QBO pressure level, date range, tropical averaging bounds, and smoothing parameters.

# Notes for multi-model QBO metrics figure: "qbo-hist-amip-summary-v4.png"

Figure "qbo-hist-amip-summary-v4.png": Periodicity, amplitude, lowest level of descent, and latitudinal width of the QBO in CMIP6 and AMIP6 models. Blue quantities show CMIP6, orange quantities show AMIP6. (a) mean periodicity for CMIP6 and AMIP6 models where whiskers extend to minimums and maximums taken across all members, CMIP6 mean is dotted, AMIP6 mean is dashed, and ERA5 is shown in black. (b) circles and squares show total (not mean) amplitudes (“TT amplitude” from Richter et al. 2020), tops of whiskers show westerly amplitudes, and bottoms of whiskers show easterly amplitudes. (c) lowest isobar the QBO descends to, the level at which the QBO Fourier amplitudes falls to 10% of its maximum (Schenzinger et al. 2017). Smaller circles and squares show ensemble members. (d) latitudinal width, at a user defined height (10 hPa here), is the full width at half amplitude maximum of a Gaussian fit to the QBO Fourier amplitude (Schenzinger et al. 2017).
