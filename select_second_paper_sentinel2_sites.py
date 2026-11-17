from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from shapely.geometry import Polygon
from shapely.ops import substring

from analyze_crossborder_gee_ndvi_20261106 import get_geometry
from build_second_paper_baseline_samples import country_geometries


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/second_paper/Sentinel2_Validation_20261114"
ARCHIVE = ROOT / "output/tables/SecondPaper_Complete_20261109"
INPUT = ROOT / "output/reports/SecondPaper_Complete_20261109/input_manifest.json"
GROUPS = ["Forest", "Cropland"]


def window(line, s):
    point = line.interpolate(s)
    a = line.interpolate(s - 750)
    b = line.interpolate(s + 750)
    tangent = np.array([b.x-a.x, b.y-a.y])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    origin = np.array([point.x, point.y])
    return Polygon([origin + u*tangent + v*normal
                    for u, v in [(-1000, -2500), (1000, -2500), (1000, 2500), (-1000, 2500)]])


def main():
    full, (russia, connector, china) = get_geometry()
    bounds = gpd.GeoSeries([full.buffer(3000)], crs=32652).to_crs(4326).total_bounds
    (cn, ru), sources = country_geometries(ROOT, bounds)
    baseline = pd.read_pickle(ARCHIVE / "analysis_inputs.pkl")
    near = baseline[baseline.distance_m.le(50)]
    starts = {"Russia": ("RU", russia, ru), "China": ("CN", china, cn)}
    sites, lines = [], []
    for country, (prefix, line, national) in starts.items():
        g = near[near.route.eq(country)]
        chosen = []
        for third in range(3):
            for group in GROUPS:
                candidates = []
                for u in range(1, int(line.length//1000)-1):
                    s = 1000*u+500
                    sample = g[g.chainage_m.between(s-1000, s+1000)]
                    count = int(sample.type.eq(group).sum())
                    if count < 120 or count/len(sample) < .65:
                        continue
                    if any(abs(s-other)<6000 for other in chosen):
                        continue
                    geom = window(line, s)
                    if geom.distance(connector) < 1000 or not national.covers(geom):
                        continue
                    mid = (third+.5)*line.length/3
                    candidates.append((abs(s-mid), -count/len(sample), -count, s, geom, len(sample), count))
                if not candidates:
                    raise RuntimeError(f"No candidate for {country} {group} third {third+1}")
                _, _, _, s, geom, n, count = min(candidates, key=lambda c:c[:4])
                chosen.append(s)
                site = f"{prefix}_{group[:2].upper()}{third+1}"
                record = dict(site=site, route=country, vegetation=group, planned_third=third+1,
                              actual_third=int(3*s/line.length)+1, chainage_m=s,
                              baseline_target_count=n, baseline_class_count=count, baseline_class_fraction=count/n)
                sites.append(dict(record, geometry=geom))
                lines.append(dict(record, geometry=substring(line, s-1000, s+1000)))
    frame = gpd.GeoDataFrame(sites, crs=32652)
    assert len(frame) == 12 and frame.site.is_unique and frame.geometry.is_valid.all()
    for i, g in enumerate(frame.geometry):
        assert all(not g.intersects(h) for h in frame.geometry.iloc[i+1:])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_file(OUT/"S2_validation_windows.gpkg", layer="windows_utm52", driver="GPKG")
    gpd.GeoDataFrame(lines, crs=32652).to_file(OUT/"S2_validation_windows.gpkg", layer="focal_centerlines_utm52", driver="GPKG", mode="a")
    frame.to_crs(4326).to_file(OUT/"S2_validation_windows.geojson",driver="GeoJSON")
    ll = gpd.GeoSeries(frame.geometry.centroid,crs=32652).to_crs(4326)
    summary = pd.DataFrame(frame.drop(columns="geometry")).assign(longitude=ll.x,latitude=ll.y)
    summary.to_csv(OUT/"S2_validation_windows.csv",encoding="utf-8-sig",index=False)
    reference = json.loads(INPUT.read_text(encoding="utf8"))["files"]["NDVI_2013"]["path"]
    with rasterio.open(reference) as ds:
        assert ds.crs.to_epsg() == 32652 and ds.res == (30,30)
        info = dict(crs=str(ds.crs), landsat_transform=list(ds.transform)[:6], landsat_source=reference,
                    country_sources=sources, window_along_m=2000, window_across_m=5000,
                    selection="For each national route third, select one forest and one cropland window "
                    "closest to the third midpoint from all eligible centres on that national route. "
                    "Rank eligible centres by proximity, class share and sample count. "
                    "Baseline vegetation share >=0.65, at least 120 baseline target pixels of focal group, "
                    "spacing >=6 km by chainage, wholly within its country mask and >1 km from the display connector.")
    (OUT/"site_selection.json").write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding="utf8")
    print(summary.to_string(index=False),flush=True)


if __name__ == "__main__":
    main()
