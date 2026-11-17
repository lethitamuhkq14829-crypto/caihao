from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
import argparse
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import shapely
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parent
RUN = "Crossborder_GEE_NDVI_20261106"
YEARS = list(range(2013, 2024))
SIDES = ["Russia", "Connector", "China"]
LAYERS = ["russia_route", "border_link_display_only", "china_route"]
EDGES_M = [50, 100, 300, 500, 2000, 10000]
BANDS = ["0–50", "50–100", "100–300", "300–500", "500–2000", "2000–10000"]
THRESHOLDS = [1, 3]
STAGES = {"baseline": slice(0, 2), "transition": slice(2, 7), "operation": slice(7, 11)}


def verify_logs(source):
    log = pd.read_csv(source / "CN_RU_annual_processing_log_ALL_2013_2023.csv")
    manifest = pd.read_csv(source / "CN_RU_Landsat_scene_manifest_ALL_2013_2023.csv")
    digest = sha256((ROOT / "data/second_paper/gee_input_20261105/Continuous_pipeline.geojson")
                    .read_bytes()).hexdigest()
    assert sorted(log.year.tolist()) == YEARS
    assert log.pipeline_source_sha256.eq(digest).all()
    assert log.crs.eq("EPSG:32652").all()
    assert log.buffer_m_each_side.eq(10000).all()
    assert log.grid_transform.eq("30,0,0,0,-30,10000020").all()
    assert log.collection.eq("LANDSAT/LC08/C02/T1_L2").all()
    assert log.composite.eq("median_of_scene_NDVI").all()
    assert log.min_valid_observations.eq(1).all()
    assert manifest.image_id.is_unique
    assert manifest.cloud_cover.lt(70).all()
    assert manifest.date.str[:4].astype(int).eq(manifest.year).all()
    assert manifest.date.str[5:7].astype(int).between(6, 9).all()
    counts = log.set_index("year").scene_count.reindex(YEARS)
    assert counts.eq(manifest.groupby("year").size().reindex(YEARS)).all()
    return digest, counts.to_numpy()


def get_geometry():
    scope = json.loads((ROOT / "data/second_paper/accepted_study_area_20261105/study_scope.json")
                       .read_text(encoding="utf-8"))
    full = gpd.read_file(scope["pipeline_gpkg"], layer=scope["analysis_layer"]).geometry.iloc[0]
    gpkg = ROOT / "output/vectors/heihe_beian_crossborder_20261104/Heihe_Beian_transboundary_study_area.gpkg"
    parts = [gpd.read_file(gpkg, layer=layer).to_crs(32652).geometry.iloc[0] for layer in LAYERS]
    assert abs(full.length - sum(p.length for p in parts)) < 0.001
    assert full.hausdorff_distance(shapely.union_all(parts)) < 0.001
    return full, parts


def raster_regions(shape, affine, domain, buffers, parts, component_buffers, full):
    labels = rasterize(
        [(poly, i) for i, poly in reversed(list(enumerate(buffers, start=1)))],
        out_shape=shape, transform=affine, dtype="uint8",
    )
    rr, cc = np.nonzero(domain)
    xx, yy = affine * (cc + 0.5, rr + 0.5)
    inside = labels[rr, cc] > 0
    if not inside.all():
        outside = shapely.points(xx[~inside], yy[~inside])
        assert np.max(shapely.distance(full, outside)) <= 10000 + 30 * np.sqrt(2) / 2 + 1
    xx, yy = xx[inside], yy[inside]
    points = shapely.points(xx, yy)
    candidates = np.vstack([shapely.contains_xy(poly, xx, yy) for poly in component_buffers])
    assert candidates.any(axis=0).all()
    group = candidates.argmax(axis=0)
    overlap = candidates.sum(axis=0) > 1
    if overlap.any():
        group[overlap] = np.vstack([shapely.distance(p, points[overlap]) for p in parts]).argmin(axis=0)
    band = labels[rr[inside], cc[inside]] - 1
    return group, band, points, inside


def get_segment_units(parts):
    geometries, offsets, tree = [], [], []
    lengths = [int(np.ceil(p.length / 1000)) for p in parts]
    for line in parts:
        coords = list(line.coords)
        edges = np.array([LineString((a, b)) for a, b in zip(coords, coords[1:])])
        lengths_m = shapely.length(edges)
        geometries.append(edges)
        offsets.append(np.r_[0, np.cumsum(lengths_m)[:-1]])
        tree.append(shapely.STRtree(edges))
    return geometries, offsets, tree, lengths


def segment_ids(points, group, units):
    geometries, offsets, tree, lengths = units
    result = np.empty(len(points), dtype=np.int32)
    for i, (edges, starts, index) in enumerate(zip(geometries, offsets, tree)):
        mask = group == i
        if not mask.any():
            continue
        point = points[mask]
        idx = index.nearest(point)
        km = ((starts[idx] + shapely.line_locate_point(edges[idx], point)) // 1000).astype(int)
        result[mask] = sum(lengths[:i]) + np.minimum(km, lengths[i] - 1)
    return result


def analyze(source, digest, scenes, full, parts):
    band_count = len(BANDS)
    zones = len(SIDES) * band_count
    domain_total = np.zeros(zones, dtype=np.int64)
    valid_total = np.zeros((len(YEARS), zones), dtype=np.int64)
    low_total = np.zeros_like(valid_total)
    obs_total = np.zeros_like(valid_total)
    units = get_segment_units(parts)
    n_units = sum(units[-1])
    cohort_count = np.zeros((len(THRESHOLDS), zones), dtype=np.int64)
    cohort_sum = np.zeros((len(THRESHOLDS), len(YEARS), zones))
    seg_count = np.zeros((len(THRESHOLDS), n_units, 2), dtype=np.int64)
    seg_sum = np.zeros((len(THRESHOLDS), len(YEARS), n_units, 2))
    buffers = [full.buffer(d, quad_segs=64) for d in EDGES_M]
    component_buffers = [p.buffer(10000.1, quad_segs=64) for p in parts]
    gee_pixels = 0
    local_only_pixels = 0
    with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024), ExitStack() as stack:
        ndvi = [stack.enter_context(rasterio.open(source / f"CN_RU_NDVI_{y}_30m.tif")) for y in YEARS]
        nobs = [stack.enter_context(rasterio.open(source / f"CN_RU_NOBS_{y}_30m.tif")) for y in YEARS]
        base = ndvi[0]
        for src in ndvi + nobs:
            assert (src.crs, src.transform, src.shape, src.count) == (base.crs, base.transform, base.shape, 1)
        assert base.crs.to_epsg() == 32652
        assert base.res == (30, 30) and base.transform.b == base.transform.d == 0
        assert (base.transform.c % 30, base.transform.f % 30) == (0, 0)
        assert all(src.nodata == -9999 and src.dtypes == ("float32",) for src in ndvi)
        assert all(src.nodata == 65535 and src.dtypes == ("uint16",) for src in nobs)
        for start in range(0, base.height, 256):
            win = Window(0, start, base.width, min(256, base.height - start))
            obs_grid = [src.read(1, window=win) for src in nobs]
            domain = obs_grid[0] != 65535
            assert all(np.array_equal(v != 65535, domain) for v in obs_grid)
            obs = np.stack([v[domain] for v in obs_grid])
            values = []
            for yi, src in enumerate(ndvi):
                v = src.read(1, window=win)
                assert np.all(v[~domain] == -9999)
                v = v[domain]
                assert np.array_equal(v != -9999, obs[yi] >= 1)
                assert np.all(np.isfinite(v)) and np.all((v == -9999) | ((v >= -1) & (v <= 1)))
                assert np.all(obs[yi] <= scenes[yi])
                values.append(v)
            if not domain.any():
                continue
            group, band, points, inside = raster_regions(
                domain.shape, base.window_transform(win), domain, buffers, parts, component_buffers, full
            )
            gee_pixels += int(domain.sum())
            local = rasterize([(buffers[-1], 1)], out_shape=domain.shape, transform=base.window_transform(win),
                              dtype="uint8").astype(bool)
            local_only_pixels += int(np.count_nonzero(local & ~domain))
            obs = obs[:, inside]
            values = [v[inside] for v in values]
            key = group * band_count + band
            domain_total += np.bincount(key, minlength=zones)
            near_ref = (band == 0) | (band == 4)
            unit = segment_ids(points[near_ref], group[near_ref], units)
            unit_key = unit * 2 + (band[near_ref] == 4)
            for yi, counts in enumerate(obs):
                valid_total[yi] += np.bincount(key[counts > 0], minlength=zones)
                low_total[yi] += np.bincount(key[(counts > 0) & (counts < 3)], minlength=zones)
                obs_total[yi] += np.bincount(key, weights=counts, minlength=zones).astype(np.int64)
            min_obs = obs.min(axis=0)
            for ti, threshold in enumerate(THRESHOLDS):
                keep = min_obs >= threshold
                keep_unit = keep[near_ref]
                k = key[keep]
                u = unit_key[keep_unit]
                cohort_count[ti] += np.bincount(k, minlength=zones)
                seg_count[ti] += np.bincount(u, minlength=n_units * 2).reshape(n_units, 2)
                for yi, v in enumerate(values):
                    cohort_sum[ti, yi] += np.bincount(k, weights=v[keep], minlength=zones)
                    seg_sum[ti, yi] += np.bincount(u, weights=v[near_ref][keep_unit],
                                                    minlength=n_units * 2).reshape(n_units, 2)
            if start % 2048 == 0:
                print(f"Checked raster rows {start + int(win.height)}/{base.height}", flush=True)
        meta = {"source": str(source), "pipeline_sha256": digest, "crs": base.crs.to_string(),
                "shape": list(base.shape), "transform": list(base.transform)[:6],
                "bounds": list(base.bounds), "raster_pairs": len(YEARS),
                "corridor_pixels": int(domain_total.sum()), "gee_corridor_pixels": gee_pixels,
                "gee_edge_pixels_excluded": gee_pixels - int(domain_total.sum()),
                "local_center_pixels_absent_from_gee": local_only_pixels, "manifest_scenes": int(sum(scenes)),
                "route_km": dict(zip(SIDES, [p.length / 1000 for p in parts]))}
    return domain_total, valid_total, low_total, obs_total, cohort_count, cohort_sum, seg_count, seg_sum, units[-1], meta


def summarize(result):
    domain, valid, low, obs, count, sums, seg_n, seg_sum, lengths, meta = result
    quality, annual, stages, segments, paired = [], [], [], [], []
    for si, side in enumerate(SIDES):
        for band, label in [(None, "0–10000"), *enumerate(BANDS)]:
            z = slice(si * len(BANDS), (si + 1) * len(BANDS)) if band is None else si * len(BANDS) + band
            n = domain[z].sum()
            for yi, year in enumerate(YEARS):
                n_valid, n_low = valid[yi, z].sum(), low[yi, z].sum()
                quality.append((side, label, year, n, n_valid, n_low, n_valid/n, (n_valid-n_low)/n, obs[yi, z].sum()/n))
            for ti, threshold in enumerate(THRESHOLDS):
                n_keep = count[ti, z].sum()
                y = np.array([sums[ti, yi, z].sum()/n_keep if n_keep else np.nan for yi in range(len(YEARS))])
                annual.extend((side, label, threshold, year, n_keep, n_keep/n, m) for year, m in zip(YEARS, y))
                a, b, c = [y[window].mean() for window in STAGES.values()]
                stages.append((side, label, threshold, n_keep, n_keep/n, a, b, c, b-a, c-a, c-b))
        for ti, threshold in enumerate(THRESHOLDS):
            sl = slice(sum(lengths[:si]), sum(lengths[:si+1]))
            ns = seg_n[ti, sl]
            sv = seg_sum[ti, :, sl]
            avg = np.divide(sv, ns, out=np.full(sv.shape, np.nan), where=ns > 0)
            gap = avg[..., 0] - avg[..., 1]
            shift = gap - gap[:2].mean(axis=0)
            eligible = (ns >= 10).all(axis=1)
            for yi, year in enumerate(YEARS):
                for seg in range(lengths[si]):
                    segments.append((side, seg, year, threshold, *ns[seg], *avg[yi, seg],
                                     gap[yi, seg], shift[yi, seg], bool(eligible[seg])))
                paired.append((side, year, threshold, int(eligible.sum()),
                               float(shift[yi, eligible].mean()) if eligible.any() else np.nan))
    return {
        "raster_quality": pd.DataFrame(quality, columns="route distance_m year corridor_pixels valid_pixels nobs_1_or_2_pixels valid_fraction nobs_ge3_fraction mean_nobs".split()),
        "annual_ndvi": pd.DataFrame(annual, columns="route distance_m min_nobs_all_years year pixels fraction_retained mean_ndvi".split()),
        "stage_ndvi": pd.DataFrame(stages, columns="route distance_m min_nobs_all_years pixels fraction_retained baseline transition operation transition_minus_baseline operation_minus_baseline operation_minus_transition".split()),
        "segment_annual_near_reference": pd.DataFrame(segments, columns="route local_km_start year min_nobs_all_years near_pixels ref_pixels near_ndvi ref_ndvi near_minus_ref baseline_adjusted_gap eligible_10px_each".split()),
        "paired_annual_gap": pd.DataFrame(paired, columns="route year min_nobs_all_years eligible_segments mean_baseline_adjusted_gap".split()),
    }, meta


def plot(tables, out):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.9), sharex=True, layout="constrained")
    for side, color, marker in [("China", "#00689D", "o"), ("Russia", "#B64F15", "s")]:
        q = tables["raster_quality"].query("route == @side and distance_m == '0–10000'")
        axes[0].plot(q.year, 100*q.nobs_ge3_fraction, label=side, color=color, marker=marker, ms=3)
        a = tables["annual_ndvi"].query("route == @side and min_nobs_all_years == 1")
        for band, style, name in [("0–50", "-", "0–50 m"), ("500–2000", "--", "500–2000 m")]:
            t = a[a.distance_m == band]
            axes[1].plot(t.year, t.mean_ndvi, label=f"{side}, {name}", color=color, ls=style, marker=marker, ms=3)
        p = tables["paired_annual_gap"].query("route == @side")
        for threshold, style in [(1, "-"), (3, "--")]:
            t = p[p.min_nobs_all_years == threshold]
            n = t.eligible_segments.iloc[0]
            axes[2].plot(t.year, t.mean_baseline_adjusted_gap, label=f"{side}, ≥{threshold} obs. ({n} segments)",
                         color=color, ls=style, marker=marker, ms=3)
    titles = ["(a) Annual observation coverage in the 10 km buffer",
              "(b) Annual NDVI at fixed pixels by distance",
              "(c) Baseline-adjusted gap relative to nearby reference pixels"]
    axes[2].axhline(0, lw=0.65, c="0.45", zorder=0)
    for ax, title, ylabel in zip(axes, titles, ["Pixels with ≥3 obs. (%)", "Mean NDVI", "NDVI gap"]):
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.legend(ncol=2, frameon=False, fontsize=7)
    axes[0].set_ylim(0, 105)
    axes[2].set(xticks=YEARS, xlabel="Year")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"Crossborder_annual_NDVI_quality_and_response.{ext}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Check the GEE export and summarize NDVI by route, distance and year.')
    parser.add_argument("--source", type=Path, default=Path.home() / "Desktop" / "NDVI")
    args = parser.parse_args()
    digest, scenes = verify_logs(args.source)
    full, parts = get_geometry()
    result = analyze(args.source, digest, scenes, full, parts)
    tables, meta = summarize(result)
    table_dir = ROOT / "output/tables" / RUN
    figure_dir = ROOT / "output/figures" / RUN
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (table_dir / "export_audit.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot(tables, figure_dir)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
