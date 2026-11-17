from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import netCDF4
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "output/tables/SecondPaper_Complete_20261109"
RAW = ROOT / "data/external/terraclimate_v1_1_second_paper_20261115/raw"
SOURCE = "https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/TERRACLIMATE_ALL/data"
YEARS = tuple(range(2013, 2024))
MONTHS = (6, 7, 8, 9)
VARIABLES = {"ppt": "mm", "tmin": "degC", "tmax": "degC"}


def request_file(job, bounds):
    variable, year = job
    name = f"TerraClimate_{variable}_{year}.nc"
    params = {
        "var": variable, **bounds, "horizStride": 1,
        "time_start": f"{year}-06-01T00:00:00Z",
        "time_end": f"{year}-09-01T00:00:00Z", "accept": "netcdf",
    }
    url = requests.Request("GET", f"{SOURCE}/{name}", params=params).prepare().url
    path = RAW / name
    if not path.exists():
        for pause in (0, 2, 5):
            time.sleep(pause)
            try:
                response = requests.get(url, timeout=(10, 35))
                response.raise_for_status()
                if not response.content.startswith((b"CDF\x01", b"CDF\x02")):
                    raise ValueError(f"Invalid NetCDF response for {url}")
                part = path.with_suffix(".part")
                part.write_bytes(response.content)
                part.replace(path)
                break
            except requests.RequestException:
                if pause == 5:
                    raise
    return {"variable": variable, "year": year, "file": path.name, "url": url,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_source(record):
    with netCDF4.Dataset(RAW / record["file"]) as ds:
        variable = ds.variables[record["variable"]]
        version = str(ds.version).upper()
        if version != "V1.1":
            raise ValueError(f"Expected v1.1 in {record['file']}, found {version}")
        units = variable.units
        accepted = {"mm"} if record["variable"] == "ppt" else {"degC", "C", "degrees Celsius"}
        if units not in accepted:
            raise ValueError(f"Unexpected units {units!r} in {record['file']}")
        dates = netCDF4.num2date(ds["time"][:], ds["time"].units)
        if [(dt.year, dt.month) for dt in dates] != [(record["year"], m) for m in MONTHS]:
            raise ValueError(f"Unexpected dates in {record['file']}: {dates}")
        if variable.dimensions != ("time", "lat", "lon"):
            raise ValueError(f"Unexpected data dimensions {variable.dimensions}")
        lat = ds["lat"][:].data.astype(float)
        lon = ds["lon"][:].data.astype(float)
        values = np.ma.asarray(variable[:], dtype="float64").filled(np.nan)
        record.update(version=version, units=units, source=str(getattr(ds, "source", "")),
                      method=str(getattr(ds, "method", "")))
    if not (np.allclose(np.diff(lat), -1 / 24, atol=1e-5)
            and np.allclose(np.diff(lon), 1 / 24, atol=1e-5)):
        raise ValueError(f"Unexpected native grid in {record['file']}")
    return lon, lat, values


def main():
    used = pd.concat([pd.read_csv(ARCHIVE / f"{country}_used_pixels.csv.gz",
                                  usecols=["longitude", "latitude"])
                      for country in ("China", "Russia")])
    bounds = dict(west=round(used.longitude.min()-0.10, 6), east=round(used.longitude.max()+0.10, 6),
                  south=round(used.latitude.min()-0.10, 6), north=round(used.latitude.max()+0.10, 6))
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(variable, year) for year in YEARS for variable in VARIABLES]
    with ThreadPoolExecutor(max_workers=6) as workers:
        records = list(workers.map(partial(request_file, bounds=bounds), jobs))
    grid = None
    for record in records:
        lon, lat, _ = read_source(record)
        if grid is None:
            grid = (lon, lat)
        if any(not np.array_equal(a, b) for a, b in zip(grid, (lon, lat))):
            raise ValueError(f"Monthly grids differ in {record['file']}")
    (RAW.parent / "source_manifest.json").write_text(
        json.dumps({"bounds": bounds, "files": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Verified {len(records)} NetCDF files on a {len(lat)} x {len(lon)} grid: {bounds}")


if __name__ == "__main__":
    main()
