"""Recompute country-level NDVI estimates from the published 1 km unit tables."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent


def main():
    operation = pd.read_csv(DATA / "operation_ndvi_by_unit.csv")
    dates = pd.read_csv(DATA / "shared_dates_ndvi_by_unit.csv")
    annual_columns = [f"D_{year}" for year in range(2020, 2024)]
    operation["operation_ndvi"] = operation[annual_columns].mean(axis=1)
    means = operation.groupby("route")["operation_ndvi"].mean()
    columns = ["near_change", "reference_change", "gap_change"]
    changes = dates.groupby("route")[columns].mean()
    np.testing.assert_allclose(means.round(4), [-0.0616, -0.1475], atol=1e-8)
    np.testing.assert_allclose(
        changes.round(4), [[-0.0656, -0.1191, 0.0535], [0.1358, 0.0407, 0.0951]], atol=1e-8
    )
    print("2020–2023 operation-period NDVI contrasts")
    print(means.round(4))
    print("2019–2023 NDVI changes on shared acquisition dates")
    print(changes.round(4))


if __name__ == "__main__":
    main()
