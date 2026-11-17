from pathlib import Path
import json

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "output/tables"
ARCHIVE = TABLES / "SecondPaper_Complete_20261109"
SEASON = TABLES / "SecondPaper_SeasonBaseline_20261111"
OUT = TABLES / "SecondPaper_Evidence_20261114"
MANIFEST = ROOT / "output/reports/SecondPaper_Complete_20261109/input_manifest.json"
YEARS = list(range(2013, 2024))
COUNTRIES = ["Russia", "China"]
COVARIATES = ["NDVI_pre", "elevation", "slope"]
REPLICATES = 2000
BLOCK_KM = 10
SEED = 20261114


def save(frame, filename):
    frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig", float_format="%.10g")


def image_audit(pixels):
    paths = {k: Path(v["path"]) for k, v in json.loads(MANIFEST.read_text())["files"].items()}
    primary_dir = paths["NDVI_2013"].parent
    log = pd.read_csv(primary_dir / "CN_RU_annual_processing_log_ALL_2013_2023.csv")
    scenes = pd.read_csv(primary_dir / "CN_RU_Landsat_scene_manifest_ALL_2013_2023.csv")
    extra = pd.read_csv(SEASON / "scene_manifest.csv")
    extra_log = pd.read_csv(SEASON / "export_logs.csv")
    assert not scenes.duplicated(["year", "image_id"]).any()
    assert not extra.duplicated(["tag", "year", "scene_id"]).any()
    assert scenes.cloud_cover.lt(70).all() and extra.cloud_cover.lt(70).all()
    counts = scenes.groupby("year").agg(scenes=("image_id", "size"), dates=("date", "nunique"))
    assert counts.scenes.equals(log.set_index("year").scene_count)
    counts["series"] = "L8_JunSep"
    secondary = extra.groupby(["tag", "year"]).agg(scenes=("scene_id", "size"), dates=("date", "nunique"))
    np.testing.assert_array_equal(secondary.scenes, extra_log.set_index(["tag", "year"]).sort_index().scene_count)
    inventory = pd.concat([counts.reset_index(), secondary.reset_index().rename(columns={"tag": "series"})], ignore_index=True)
    save(inventory, "annual_scene_inventory.csv")
    samples = pd.concat([pixels[pixels.route == route].sample(24, random_state=SEED) for route in COUNTRIES])
    coordinates = list(zip(samples.x, samples.y))
    grid = None
    rows = []
    for year in YEARS:
        for variable, nodata in [("NDVI", -9999), ("NOBS", 65535)]:
            name = f"{variable}_{year}"
            with rasterio.open(paths[name]) as ds:
                signature = (ds.crs, ds.shape, ds.transform)
                if grid is None:
                    grid = signature
                assert signature == grid and ds.crs.to_epsg() == 32652
                assert ds.res == (30, 30) and ds.nodata == nodata and ds.count == 1
                values = np.array([item[0] for item in ds.sample(coordinates)], dtype=float)
                if variable == "NDVI":
                    values[values == nodata] = np.nan
                np.testing.assert_allclose(values, samples[name].to_numpy(float), atol=1e-7, rtol=0)
                rows.append(dict(variable=variable, year=year, source=paths[name].name,
                                 epsg=32652, columns=ds.width, rows=ds.height, nodata=nodata,
                                 pixel_size_m=30, x_origin=ds.transform.c, y_origin=ds.transform.f,
                                 checked_pixels=len(samples)))
    save(pd.DataFrame(rows), "primary_raster_audit.csv")
    return inventory


def quality_and_matches(pixels):
    pairs = pd.read_csv(ARCHIVE / "pairs/base.csv.gz")
    assert pairs.target_id.is_unique and len(pairs) == 46338
    assert pixels.pixel_id.is_unique
    keyed = pixels.set_index("pixel_id")
    targets = keyed.loc[pairs.target_id].reset_index()
    control_ids = pairs[["control1_id", "control2_id", "control3_id"]].to_numpy()
    controls = keyed.loc[control_ids.ravel()]
    assert (np.diff(np.sort(control_ids, axis=1), axis=1) != 0).all()
    for col in ["route", "unit", "lc2013"]:
        assert (targets[col].to_numpy()[:, None] == controls[col].to_numpy().reshape(-1, 3)).all()
    assert targets.distance_m.between(0, 50).all() and controls.distance_m.between(500, 2000).all()
    obscols = [f"NOBS_{year}" for year in YEARS]
    valid_ndvi = np.isfinite(targets[[f"NDVI_{y}" for y in YEARS]].to_numpy())
    valid_ndvi &= np.isfinite(controls[[f"NDVI_{y}" for y in YEARS]].to_numpy().reshape(-1, 3, 11)).all(axis=1)
    obs = np.minimum(targets[obscols].to_numpy(), controls[obscols].to_numpy().reshape(-1, 3, 11).min(axis=1))
    assert (obs < 65535).all()
    valid = {n: (obs >= n) & valid_ndvi for n in [1, 3]}
    for n, mask in valid.items():
        targets[f"complete_{n}"] = mask.all(axis=1)
    rows = []
    for country in COUNTRIES:
        take = targets.route.eq(country).to_numpy()
        for n in valid:
            for j, year in enumerate(YEARS):
                rows.append(dict(route=country, threshold=n, year=year, matched=int(take.sum()),
                                 valid=int(valid[n][take, j].sum()), percentage=100 * valid[n][take, j].mean()))
    cover = pd.DataFrame(rows)
    reference = pd.read_csv(ARCHIVE / "annual_coverage.csv").query("analysis == 'base'")
    compare = cover.query("threshold == 1").merge(reference, on=["route", "year"], validate="one_to_one")
    assert (compare.valid_x == compare.valid_y).all()
    save(cover, "annual_four_pixel_coverage.csv")
    cohort = pd.read_csv(ARCHIVE / "base_complete_targets.csv.gz")
    assert set(cohort.pixel_id) == set(targets.loc[targets.complete_1, "pixel_id"])
    cohort = cohort.merge(targets[["pixel_id", "complete_3"]], on="pixel_id", validate="one_to_one")
    return targets, controls, cohort


def weighted_moments(frame, col):
    moments = frame.assign(square=frame[col] ** 2).groupby("unit")[[col, "square"]].mean()
    mean, second = moments.mean()
    return mean, second - mean ** 2


def balance_profiles(pixels, targets, controls):
    rows = []
    for country in COUNTRIES:
        keep = targets.route.eq(country).to_numpy() & targets.complete_1.to_numpy()
        t = targets.loc[keep]
        c = controls.iloc[np.repeat(keep, 3)]
        pool = pixels[pixels.route.eq(country) & pixels.unit.isin(t.unit)]
        unmatched_t = pool[pool.distance_m.between(0, 50)]
        unmatched_c = pool[pool.distance_m.between(500, 2000)]
        for phase, near, ref in [("Before matching", unmatched_t, unmatched_c), ("Complete matches", t, c)]:
            assert set(near.unit) == set(ref.unit) == set(t.unit)
            for col in COVARIATES:
                nm, nv = weighted_moments(near, col)
                rm, rv = weighted_moments(ref, col)
                rows.append(dict(route=country, sample=phase, covariate=col, n_units=t.unit.nunique(),
                                 n_targets=len(near), reference_uses=len(ref), target_mean=nm, reference_mean=rm,
                                 smd=(nm-rm) / np.sqrt((nv+rv)/2)))
    save(pd.DataFrame(rows), "unit_weighted_balance.csv")


def bootstrap_units(units):
    values = units.to_numpy()
    blocks, inv = np.unique(units.index.to_numpy() // BLOCK_KM, return_inverse=True)
    sums = np.zeros((len(blocks), values.shape[1]))
    np.add.at(sums, inv, values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(blocks), (REPLICATES, len(blocks)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)[:, None]
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return values.mean(0), lo, hi, len(blocks)


def tidy_bootstrap(units):
    means, lows, highs, n_blocks = bootstrap_units(units)
    return pd.DataFrame(dict(metric=units.columns, mean=means, low=lows, high=highs,
                             n_units=len(units), n_blocks=n_blocks))


def annual_components(cohort):
    records = []
    averages = []
    for country, subset in cohort.groupby("route"):
        result = pd.DataFrame({"unit": subset.unit})
        for kind in ["near", "ref"]:
            baseline = subset[[f"{kind}_2013", f"{kind}_2014"]].mean(axis=1)
            for year in YEARS:
                result[f"{kind}_{year}"] = subset[f"{kind}_{year}"] - baseline
        for year in YEARS:
            result[f"adjusted_{year}"] = result[f"near_{year}"] - result[f"ref_{year}"]
            np.testing.assert_allclose(result[f"adjusted_{year}"], subset[f"D_{year}"], atol=1e-7, rtol=0)
        units = result.groupby("unit").mean()
        curves = tidy_bootstrap(units)
        curves[["component", "year"]] = curves.metric.str.split("_", expand=True)
        records.append(curves.assign(route=country, n_targets=len(subset)))
        stage = {}
        for role in ["near", "ref", "adjusted"]:
            stage[f"{role}_operation"] = units[[f"{role}_{y}" for y in range(2020, 2024)]].mean(axis=1)
            stage[f"{role}_rebound"] = units[f"{role}_2023"] - units[f"{role}_2019"]
        averages.append(tidy_bootstrap(pd.DataFrame(stage)).assign(route=country, n_targets=len(subset)))
    save(pd.concat(records, ignore_index=True), "annual_components.csv")
    save(pd.concat(averages, ignore_index=True), "phase_and_rebound_components.csv")


def observation_sensitivity(cohort):
    outcomes = cohort.assign(operation=cohort[[f"D_{y}" for y in range(2020, 2024)]].mean(axis=1))
    archive = pd.read_csv(ARCHIVE / "effect_summaries.csv")
    records, period = [], []
    for country, subset in outcomes.groupby("route"):
        stricter = subset[subset.complete_3]
        shared = subset[subset.unit.isin(stricter.unit)]
        original = subset.groupby("unit").operation.mean()
        same_units = shared.groupby("unit").operation.mean()
        strict_units = stricter.groupby("unit").operation.mean()
        assert same_units.index.equals(strict_units.index)
        aligned = pd.DataFrame(dict(original=same_units, strict=strict_units, difference=strict_units-same_units))
        count = dict(original=len(shared), strict=len(stricter), difference=len(stricter))
        table = tidy_bootstrap(aligned).assign(route=country)
        table["n_targets"] = table.metric.map(count)
        records.extend([tidy_bootstrap(original.to_frame("full")).assign(route=country, n_targets=len(subset)), table])
        for branch, expected in [("base", original.mean()), ("NOBS3", strict_units.mean())]:
            ref = archive.query("route == @country and analysis == @branch and type == 'All' and metric == 'operation' and block_km == 10")
            assert len(ref) == 1 and abs(ref.iloc[0]["mean"] - expected) < 1e-7
        for branch, metric in [("base", "operation"), ("base", "operation_2020_2022"), ("end2022", "operation_2020_2022")]:
            period.append(archive.query("route == @country and analysis == @branch and type == 'All' and metric == @metric and block_km == 10"))
    save(pd.concat(records, ignore_index=True), "observation_threshold_common_units.csv")
    save(pd.concat(period, ignore_index=True), "operation_window_support.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pixels = pd.read_pickle(ARCHIVE / "analysis_inputs.pkl")
    image_audit(pixels)
    targets, controls, cohort = quality_and_matches(pixels)
    balance_profiles(pixels, targets, controls)
    annual_components(cohort)
    observation_sensitivity(cohort)
    record = dict(seed=SEED, repetitions=REPLICATES, block_km=BLOCK_KM, matched_targets=len(targets),
                  complete_targets=len(cohort), primary_checks=22 * 48, countries=COUNTRIES)
    (OUT / "verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
