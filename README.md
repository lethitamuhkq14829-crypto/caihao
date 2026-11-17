# Vegetation greenness along a China–Russia gas pipeline

Analysis code and 1 km unit statistics for the manuscript *Persistent near-line vegetation greenness deficits along a China–Russia gas pipeline*.

## Replicate the reported point estimates

Run the following from the repository root.

```sh
python -m pip install -r requirements-replay.txt
python replay_published_estimates.py
```

The script checks the 2020–2023 operation-period NDVI contrasts and the 2019–2023 comparison using acquisition dates shared by each matched set. Both calculations give equal weight to the 1 km units within each national section.

| Location | Content |
| --- | --- |
| Earth Engine scripts (`.js`) | Annual Landsat 8 NDVI and valid-scene counts, July–August Landsat 8, the early Landsat 7 baseline, and baseline land cover and elevation |
| Python scripts (`.py`) | Sample matching, cross-country comparison, seasonal sensitivity, weather analysis, Sentinel-2 validation and the shared-date Landsat analysis |
| [Operation-period data](operation_ndvi_by_unit.csv) | Annual baseline-adjusted near-line minus reference NDVI contrasts, 2020–2023 |
| [Shared-date data](shared_dates_ndvi_by_unit.csv) | Changes from 2019 to 2023 in near-line NDVI, reference NDVI and their difference |

The two CSVs contain 245 Chinese units and 178 Russian units. In each CSV, `route` identifies the national section, `unit` is a sequential row identifier within the section, and `matched_sets` is the number of contributing target–reference sets. The tables support replication of the reported point estimates. Computing uncertainty intervals and repeating pixel-level matching require the archived inputs, including the exact study geometries and rasters.

## Rerun image processing

Load a JavaScript file in the Google Earth Engine Code Editor. Set the Earth Engine asset identifiers at the top of that script to use the respective line or buffer. The two seasonal scripts require the 2.1 km corridor and the covariate script requires the 10 km corridor. For an exact rerun, use the archived assets and grid. To rerun the full Python analyses, arrange the extracted raster and other intermediate inputs as indicated in the scripts. Some stages take project paths as arguments.

Public source datasets are listed in the manuscript and include [USGS Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2), [USGS Landsat 7](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2), [Planetary Computer Sentinel-2](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a), and [ESA CCI Permafrost v5.0](https://catalogue.ceda.ac.uk/uuid/d235665772ec4b558e9a89ac85595e71/).
