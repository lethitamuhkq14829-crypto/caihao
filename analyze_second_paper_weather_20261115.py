"""Compare seasonal weather slopes of pipeline targets and matched references."""

import calendar
import json

import numpy as np
import pandas as pd

from fetch_second_paper_terraclimate_20261115 import ARCHIVE, MONTHS, RAW, YEARS, ROOT, read_source

OUTPUT = ROOT / "output/tables/SecondPaper_Weather_20261115"
FEATURES = ["temperature_c", "precipitation_100mm"]
RESPONSES = ["near", "reference", "gap"]
PREDICTORS = [*FEATURES, *RESPONSES]
CONTROLS = ["control1_id", "control2_id", "control3_id"]
NDVI_COLS = [f"{prefix}_{year}" for prefix in ("near", "ref", "D") for year in YEARS]


def load_weather():
    records = json.loads((RAW.parent / "source_manifest.json").read_text(encoding="utf-8"))
    values = {}
    for record in records["files"]:
        lon, lat, array = read_source(record)
        values[record["year"], record["variable"]] = array
    rows = []
    for year in YEARS:
        days = [calendar.monthrange(year, month)[1] for month in MONTHS]
        low = values[year, "tmin"]
        high = values[year, "tmax"]
        rain = values[year, "ppt"]
        if np.any(high < low) or np.any(rain < 0):
            raise ValueError(f"Inconsistent temperature or precipitation in {year}")
        monthly_temp = (low + high) / 2
        temp = np.average(monthly_temp, weights=days, axis=0)
        annual_rain = rain.sum(axis=0)
        rows.append(pd.DataFrame({
            "year": year, "cell": np.arange(len(lat) * len(lon)),
            "cell_lon": np.tile(lon, len(lat)), "cell_lat": np.repeat(lat, len(lon)),
            "temperature_c": temp.ravel(), "precipitation_mm": annual_rain.ravel(),
        }))
    weather = pd.concat(rows, ignore_index=True)
    weather["precipitation_100mm"] = weather.precipitation_mm / 100
    return weather, lon, lat


def assign_cell(points, lon, lat):
    # Native grid centres. Longitude increases eastward and latitude decreases southward.
    x, y = points.longitude.to_numpy(), points.latitude.to_numpy()
    half = 1 / 48
    outside = ((x < lon[0] - half) | (x > lon[-1] + half)
               | (y < lat[-1] - half) | (y > lat[0] + half))
    if not np.isfinite(np.column_stack((x, y))).all() or outside.any():
        raise ValueError("A matched pixel is outside the downloaded grid")
    col = np.searchsorted((lon[1:] + lon[:-1]) / 2, x)
    row = np.searchsorted(-(lat[1:] + lat[:-1]) / 2, -y)
    return row * len(lon) + col


def target_data(weather, lon, lat):
    data = pd.read_csv(ARCHIVE / "base_complete_targets.csv.gz", usecols=[
        "route", "unit", "pixel_id", "longitude", "latitude", "chainage_m", *NDVI_COLS
    ])
    data["cell"] = assign_cell(data, lon, lat)
    pairs = pd.read_csv(ARCHIVE / "pairs/base.csv.gz", usecols=["target_id", *CONTROLS])
    data = data.merge(pairs, left_on="pixel_id", right_on="target_id", validate="one_to_one")
    if len(data) != 45544:
        raise ValueError(f"Expected all 45,544 sets, found {len(data)}")
    pixels = pd.concat([pd.read_csv(ARCHIVE / f"{c}_used_pixels.csv.gz",
                                   usecols=["pixel_id", "longitude", "latitude"])
                        for c in ("China", "Russia")], ignore_index=True)
    if pixels.pixel_id.duplicated().any():
        raise ValueError("Pixel coordinates must have unique IDs")
    cell_by_pixel = pd.Series(assign_cell(pixels, lon, lat), index=pixels.pixel_id)
    members = np.column_stack([data.cell, *(cell_by_pixel.loc[data[c]].to_numpy() for c in CONTROLS)])
    data["same_cell"] = (members == members[:, :1]).all(axis=1)
    sampled = weather[weather.cell.isin(np.unique(members))]
    if not np.isfinite(sampled[FEATURES]).all().all():
        raise ValueError("A cell containing a target or reference lacks complete climate observations")
    near = data[[f"near_{y}" for y in YEARS]].to_numpy()
    reference = data[[f"ref_{y}" for y in YEARS]].to_numpy()
    gap = near - reference
    adjusted = data[[f"D_{y}" for y in YEARS]].to_numpy()
    if not np.allclose(gap - gap[:, :2].mean(axis=1, keepdims=True), adjusted, atol=5e-7):
        raise ValueError("Recomputed baseline-centred gaps differ from the archived contrasts")
    return data, sampled


def cell_panel(data, weather, min_targets):
    eligible = data.groupby("cell").filter(lambda rows: len(rows) >= min_targets)
    averages = eligible.groupby(["cell", "unit"])[NDVI_COLS].mean().groupby("cell").mean()
    cell_info = eligible.groupby("cell").agg(
        chainage_m=("chainage_m", "median"), targets=("pixel_id", "size"), units=("unit", "nunique")
    )
    rows = []
    for year in YEARS:
        annual = averages[[f"near_{year}", f"ref_{year}"]].set_axis(["near", "reference"], axis=1)
        annual["gap"] = annual.near - annual.reference
        rows.append(annual.join(cell_info).assign(year=year).reset_index())
    panel = pd.concat(rows, ignore_index=True).merge(weather, on=["cell", "year"], validate="one_to_one")
    if len(panel) != len(cell_info) * len(YEARS) or not np.isfinite(panel[PREDICTORS]).all().all():
        raise ValueError("Cell-year panel must be complete")
    return panel.sort_values(["cell", "year"]), eligible


def estimate(values, phase):
    # With a balanced panel, subtract cell-by-period and year means and restore
    # the period mean. A single period yields ordinary cell and year effects.
    demeaned = values.copy()
    for stage in np.unique(phase):
        mask = phase == stage
        section = values[:, mask]
        demeaned[:, mask] = (section - section.mean(axis=0, keepdims=True)
                             - section.mean(axis=1, keepdims=True)
                             + section.mean(axis=(0, 1), keepdims=True))
    predictors = demeaned[:, :, :2].reshape(-1, 2)
    response = demeaned[:, :, 2:].reshape(-1, 3)
    fitted, _, rank, _ = np.linalg.lstsq(predictors, response, rcond=None)
    if rank != 2:
        return np.full((2, 3), np.nan)
    return fitted


def summarize(route, scenario, panel, targets, length_km=10, per_stage=False, draws=2000):
    values = panel[PREDICTORS].to_numpy().reshape(-1, len(YEARS), len(PREDICTORS))
    phase = np.digitize(YEARS, [2015, 2020]) if per_stage else np.zeros(len(YEARS), dtype=int)
    cells = panel.drop_duplicates("cell")
    _, group = np.unique(cells.chainage_m.to_numpy() // (length_km * 1000), return_inverse=True)
    groups = [np.flatnonzero(group == i) for i in np.unique(group)]
    rng = np.random.default_rng(20261115)
    fits = []
    for _ in range(draws):
        rows = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        cols = rng.integers(len(YEARS), size=len(YEARS))
        fits.append(estimate(values[rows][:, cols], phase[cols]))
    fits = np.stack(fits)
    valid = np.isfinite(fits).all(axis=(1, 2))
    if valid.mean() < 0.98:
        raise ValueError(f"{route} {scenario}: rank deficiency in bootstrap draws")
    low, high = np.quantile(fits[valid], [0.025, 0.975], axis=0)
    point = estimate(values, phase)
    if not np.allclose(point[:, 0] - point[:, 1], point[:, 2], atol=1e-11):
        raise ValueError("Point estimates must satisfy near minus reference equals gap")
    omitted = []
    for drop_index in range(len(YEARS)):
        keep = np.arange(len(YEARS)) != drop_index
        omitted.append(estimate(values[:, keep], phase[keep]))
    rows = [
        dict(route=route, scenario=scenario, block_km=length_km, predictor=feature, outcome=outcome,
             slope=point[p, q], low95=low[p, q], high95=high[p, q],
             leave_one_year_out_min=np.min(omitted, axis=0)[p, q],
             leave_one_year_out_max=np.max(omitted, axis=0)[p, q], targets=len(targets),
             units=targets.unit.nunique(), cells=len(cells), blocks=len(groups), years=len(YEARS),
             bootstrap_draws=valid.sum())
        for p, feature in enumerate(FEATURES) for q, outcome in enumerate(RESPONSES)
    ]
    return rows


def main():
    weather, lon, lat = load_weather()
    targets, sampled = target_data(weather, lon, lat)
    rows, panels, coverage = [], [], []
    for route, subset in targets.groupby("route"):
        share = subset[subset.same_cell]
        coverage.append(dict(route=route, total=len(subset), total_units=subset.unit.nunique(),
                             same_cell=len(share), same_cell_units=share.unit.nunique(),
                             target_cells=subset.cell.nunique(), same_cell_target_cells=share.cell.nunique()))
        main_panel, main_sets = cell_panel(share, sampled, 10)
        panels.append(main_panel.assign(route=route))
        rows.extend(summarize(route, "primary", main_panel, main_sets))
        rows.extend(summarize(route, "20km_blocks", main_panel, main_sets, length_km=20))
        rows.extend(summarize(route, "cell_by_period", main_panel, main_sets, per_stage=True))
        small_panel, small_sets = cell_panel(share, sampled, 5)
        rows.extend(summarize(route, "at_least_5_sets", small_panel, small_sets))
        all_panel, all_sets = cell_panel(subset, sampled, 10)
        rows.extend(summarize(route, "target_cell_proxy", all_panel, all_sets))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUTPUT / "matched_climate_coverage.csv", index=False)
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "weather_associations.csv", index=False, float_format="%.9f")
    pd.concat(panels).to_csv(OUTPUT / "same_cell_weather_panel.csv", index=False, float_format="%.9f")
    sampled.drop(columns="precipitation_100mm").to_csv(
        OUTPUT / "native_cells_growing_season_weather.csv", index=False, float_format="%.6f"
    )
    print(pd.DataFrame(coverage).to_string(index=False))
    print(results.query("scenario == 'primary' or outcome == 'gap'").to_string(index=False))


if __name__ == "__main__":
    main()
