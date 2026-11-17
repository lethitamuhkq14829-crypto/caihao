"""Extract per-scene Landsat 8 NDVI at the original matched pixel centres."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
import requests

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "output/tables/SecondPaper_Complete_20261109"
OUT = ROOT / "data/second_paper/Landsat_Common_Dates_20261116"
API = "https://planetarycomputer.microsoft.com/api"
ASSETS = ("red", "nir08", "qa_pixel", "qa_radsat")
YEARS = (2019, 2023)


def catalog_item(record):
    parts = record.product_id.split("_")
    item_id = "_".join(parts[:4] + parts[5:])
    url = f"{API}/stac/v1/collections/landsat-c2-l2/items/{item_id}"
    response = requests.get(url, timeout=22)
    response.raise_for_status()
    item = response.json()
    meta = item["properties"]
    if (meta["landsat:scene_id"] != record.scene_id
            or meta["datetime"][:10] != record.date
            or meta["landsat:collection_category"] != "T1"):
        raise ValueError(f"Catalog scene or date differs from original manifest: {item_id}")
    hrefs = [item["assets"][key]["href"] for key in ASSETS]
    # Input archive tracks the exact product, including the processing timestamp.
    if not all(record.product_id + "_" in href for href in hrefs):
        raise ValueError(f"Processing version differs: {record.product_id}")
    return item_id, hrefs


@lru_cache(maxsize=16)
def sign_container(account, container):
    response = requests.get(f"{API}/sas/v1/token/{account}/{container}", timeout=20)
    response.raise_for_status()
    return response.json()["token"]


def signed(url):
    blob = urlsplit(url)
    return url + "?" + sign_container(blob.hostname.split(".")[0], blob.path.split("/")[1])


def sample(hrefs, x, y):
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="20",
                      GDAL_HTTP_CONNECTTIMEOUT="8", GDAL_HTTP_MAX_RETRY="2",
                      GDAL_HTTP_RETRY_DELAY="1"):
        with ExitStack() as stack:
            images = [stack.enter_context(rasterio.open(signed(href))) for href in hrefs]
            main = images[0]
            if any((s.crs, s.shape, s.transform) != (main.crs, main.shape, main.transform)
                   for s in images):
                raise ValueError("Within-scene assets have inconsistent grids")
            if main.crs.to_epsg() != 32652:
                x, y = Transformer.from_crs(32652, main.crs, always_xy=True).transform(x, y)
            cols, rows = ~main.transform * (x, y)
            rr, cc = np.floor(rows).astype(int), np.floor(cols).astype(int)
            use = np.flatnonzero((rr >= 0) & (rr < main.height) & (cc >= 0) & (cc < main.width))
            result = np.full(len(x), np.nan, np.float32)
            if not len(use):
                return result
            # Run reads by GDAL internal block, sharing HTTP ranges between points.
            bh, bw = main.block_shapes[0]
            ncol = (main.width + bw - 1) // bw
            blocks = (rr[use] // bh) * ncol + cc[use] // bw
            order = np.argsort(blocks)
            ids = np.split(use[order], np.flatnonzero(np.diff(blocks[order])) + 1)
            for indices in ids:
                r0, c0 = (rr[indices[0]] // bh) * bh, (cc[indices[0]] // bw) * bw
                window = Window(c0, r0, min(bw, main.width - c0), min(bh, main.height - r0))
                rc = rr[indices] - r0, cc[indices] - c0
                qa = images[2].read(1, window=window)[rc]
                valid = (qa & 63) == 0
                if not valid.any():
                    continue
                radsat = images[3].read(1, window=window)[rc]
                valid &= radsat == 0
                if not valid.any():
                    continue
                red, nir = [s.read(1, window=window)[rc] * 0.0000275 - 0.2 for s in images[:2]]
                valid &= (red >= 0) & (nir >= 0) & ((red + nir) > 0)
                result[indices[valid]] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
            return result


def run_scene(args):
    row, x, y, pixels = args
    cache = OUT / "scenes" / f"{row.product_id}.npz"
    if cache.exists():
        return row.product_id
    for attempt in range(2):
        try:
            item_id, hrefs = catalog_item(row)
            values = sample(hrefs, x, y)
            temp = cache.with_suffix(".partial.npz")
            np.savez_compressed(temp, ndvi=values, pixel_id=pixels, date=row.date,
                                item=item_id, unsigned_assets=np.array(hrefs))
            temp.replace(cache)
            print(f"{item_id} {np.isfinite(values).sum()} valid pixels", flush=True)
            return row.product_id
        except Exception as error:
            # Some GDAL errors include signed URLs. Avoid printing credentials.
            print(f"{row.product_id} attempt {attempt + 1}: {type(error).__name__}", flush=True)
    return None


def points():
    base = pd.read_csv(ARCHIVE / "base_complete_targets.csv.gz", usecols=["pixel_id"])
    pairs = pd.read_csv(ARCHIVE / "pairs/base.csv.gz")
    pairs = pairs[pairs.target_id.isin(base.pixel_id)]
    assert len(pairs) == len(base)
    needed = np.unique(pairs[["target_id", "control1_id", "control2_id", "control3_id"]].to_numpy())
    coordinates = pd.concat([
        pd.read_csv(ARCHIVE / f"{c}_used_pixels.csv.gz", usecols=["pixel_id", "x", "y"])
        for c in ("China", "Russia")
    ]).set_index("pixel_id").loc[needed]
    assert coordinates.index.is_unique
    return coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.year.isin(YEARS)].copy()
    assert manifest.product_id.is_unique and manifest.groupby("year").size().to_dict() == {2019: 32, 2023: 19}
    coords = points()
    (OUT / "scenes").mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "common_date_scene_manifest.csv", index=False)
    jobs = [(row, coords.x.to_numpy(), coords.y.to_numpy(), coords.index.to_numpy())
            for row in manifest.itertuples(index=False)]
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        obtained = list(workers.map(run_scene, jobs))
    missing = [row.product_id for row, result in zip(manifest.itertuples(), obtained) if result is None]
    (OUT / "extraction_status.json").write_text(
        json.dumps({"pixels": len(coords), "requested": len(obtained), "missing": missing}, indent=2),
        encoding="utf-8",
    )
    if missing:
        raise SystemExit(f"{len(missing)} Landsat scenes are missing")


if __name__ == "__main__":
    main()
