import argparse
import json
import warnings

import numpy as np
import pandas as pd

from extract_second_paper_landsat_dates_20261116 import ARCHIVE, OUT, ROOT, YEARS

RESULTS = ROOT / "output/tables/SecondPaper_Submission_20261116/common_dates"
MEMBERS = ["target_id", "control1_id", "control2_id", "control3_id"]
OUTCOMES = ["near_change", "reference_change", "gap_change"]


def median(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def read_scene_matrices(pairs):
    manifest = pd.read_csv(OUT / "common_date_scene_manifest.csv")
    expected = {row.product_id for row in manifest.itertuples()}
    paths = {p.stem: p for p in (OUT / "scenes").glob("*.npz") if p.stem in expected}
    if set(paths) != expected:
        raise FileNotFoundError(f"Missing required Landsat scenes: {sorted(expected - set(paths))}")
    with np.load(next(iter(paths.values()))) as f:
        pixel_ids = f["pixel_id"]
    ids = pairs[MEMBERS].to_numpy()
    idx = np.searchsorted(pixel_ids, ids)
    assert np.array_equal(pixel_ids[idx], ids)
    archive = pd.concat([pd.read_csv(ARCHIVE / f"{c}_used_pixels.csv.gz", usecols=[
        "pixel_id", *(f"{v}_{y}" for v in ("NDVI", "NOBS") for y in YEARS)
    ]) for c in ("Russia", "China")]).set_index("pixel_id").loc[pixel_ids]
    synchronized, independent, dates_per_target, qa = {}, {}, {}, []
    for year, subset in manifest.groupby("year"):
        scenes, daily = [], []
        for date, day in subset.groupby("date"):
            values = []
            for key in day.product_id:
                with np.load(paths[key]) as f:
                    assert np.array_equal(f["pixel_id"], pixel_ids) and f["date"].item() == date
                    values.append(f["ndvi"])
            scenes.extend(values)
            daily.append(median(np.stack(values)) if len(values) > 1 else values[0])
        scenes = np.stack(scenes)
        scenes_med = median(scenes)
        src = archive[f"NDVI_{year}"].to_numpy()
        valid = np.isfinite(src) & np.isfinite(scenes_med)
        differences = np.abs(scenes_med[valid] - src[valid])
        qa.append(dict(year=int(year), scenes=len(scenes), dates=len(daily),
                       nobs_equal_fraction=float(
                           (np.isfinite(scenes).sum(axis=0) == archive[f"NOBS_{year}"]).mean()
                       ),
                       comparable_pixels=int(valid.sum()),
                       ndvi_median_absolute_difference=float(np.median(differences)),
                       ndvi_p95_absolute_difference=float(np.quantile(differences, .95))))
        daily_members = np.stack(daily)[:, idx]
        shared = np.isfinite(daily_members).all(axis=2)
        dates_per_target[year] = shared.sum(axis=0)
        synchronized[year] = median(np.where(shared[:, :, None], daily_members, np.nan))
        independent[year] = median(daily_members)
    return synchronized, independent, dates_per_target, qa


def measurements(change):
    bg = change[:, 1:].mean(axis=1)
    return np.column_stack([change[:, 0], bg, change[:, 0] - bg])


def summary(country, threshold, frame, columns, block_km=10):
    units = frame.groupby("unit")[columns].mean()
    group = units.groupby(units.index // block_km)
    sums = group.sum().to_numpy()
    counts = group.size().to_numpy()
    rng = np.random.default_rng(20261116)
    draws = rng.integers(0, len(counts), (2000, len(counts)))
    sims = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]
    low, high = np.quantile(sims, [.025, .975], axis=0)
    def row(col, j):
        branch, measure = col.split("::")
        return dict(route=country, min_common_dates=threshold, series=branch, outcome=measure,
                    estimate=units.loc[:, col].mean(), low95=low[j], high95=high[j],
                    targets=len(frame), units=len(units), blocks=len(counts), block_km=block_km)
    return [row(col, j) for j, col in enumerate(columns)]


def main():
    args = argparse.ArgumentParser()
    args.parse_args()
    base = pd.read_csv(ARCHIVE / "base_complete_targets.csv.gz", usecols=[
        "pixel_id", "route", "unit", *(f"{v}_{y}" for v in ("near", "ref") for y in YEARS)
    ])
    pairs = pd.read_csv(ARCHIVE / "pairs/base.csv.gz", usecols=MEMBERS)
    frame = base.merge(pairs, left_on="pixel_id", right_on="target_id", validate="one_to_one")
    assert len(frame) == 45544
    sync, independent, dates, qa = read_scene_matrices(frame)
    frame[F"shared_dates_{YEARS[0]}"] = dates[YEARS[0]]
    frame[F"shared_dates_{YEARS[1]}"] = dates[YEARS[1]]
    sync_change = measurements(sync[2023] - sync[2019])
    independent_change = measurements(independent[2023] - independent[2019])
    near = frame.near_2023 - frame.near_2019
    ref = frame.ref_2023 - frame.ref_2019
    values = {
        "archive": np.column_stack([near, ref, near - ref]),
        "independent_dates": independent_change,
        "same_dates": sync_change,
        "same_minus_independent": sync_change - independent_change,
    }
    all_cols = []
    for branch, matrix in values.items():
        names = [f"{branch}::{name}" for name in OUTCOMES]
        frame[names] = matrix
        all_cols.extend(names)
    output, counts = [], []
    for route, sub in frame.groupby("route"):
        output.extend(summary(route, 0, sub, all_cols[:3]))
        for minimum in (1, 2, 3):
            retained = sub[sub.shared_dates_2019.ge(minimum) & sub.shared_dates_2023.ge(minimum)]
            counts.append(dict(route=route, min_dates=minimum, targets=len(retained),
                               units=retained.unit.nunique(), base_targets=len(sub),
                               base_units=sub.unit.nunique()))
            if len(retained):
                assert np.isfinite(retained[all_cols]).all().all()
                output.extend(summary(route, minimum, retained, all_cols))
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame[["route", "unit", *MEMBERS, "shared_dates_2019", "shared_dates_2023", *all_cols]].to_csv(
        RESULTS / "matched_targets_common_dates.csv.gz", index=False, float_format="%.8f")
    output = pd.DataFrame(output)
    output.to_csv(RESULTS / "same_date_component_changes.csv", index=False, float_format="%.9f")
    pd.DataFrame(counts).to_csv(RESULTS / "same_date_coverage.csv", index=False)
    (RESULTS / "landsat_archive_replay_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(qa)
    print(pd.DataFrame(counts).to_string(index=False))
    print(output.query("min_common_dates == 1").to_string(index=False))


if __name__ == "__main__":
    main()
