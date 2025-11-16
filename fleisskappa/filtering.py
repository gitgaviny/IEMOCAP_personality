#!/usr/bin/env python3
import pandas as pd
import numpy as np

INPUT_CSV = "path/to/ocean_6raters.csv"
OUTPUT_CSV = "ocean_5raters.csv"

RATER_COLS_6 = [f"rater{i}" for i in range(1, 7)]
RATER_COLS_5 = [f"rater{i}" for i in range(1, 6)]


def keep_five_raters(row):
    """Keep 5 raters by dropping the one farthest from the mean."""
    # get 6 ratings as float array
    vals = row[RATER_COLS_6].to_numpy(dtype=float)

    # mean of 6 ratings
    mean_val = float(np.mean(vals))

    # absolute distance to mean
    diffs = np.abs(vals - mean_val)

    # index (0~5) of the value farthest from the mean
    drop_idx = int(diffs.argmax())

    # keep the remaining 5 values in the original order
    kept_vals = [v for i, v in enumerate(vals) if i != drop_idx]

    # return as a Series: rater1 ~ rater5
    return pd.Series(kept_vals, index=RATER_COLS_5)


def main():
    df = pd.read_csv(INPUT_CSV)

    # check required rater columns
    for c in RATER_COLS_6:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # apply row-wise to generate new rater1~rater5
    new_raters = df.apply(keep_five_raters, axis=1)

    # drop old rater1~rater6, append new rater1~rater5
    df_out = pd.concat(
        [df.drop(columns=RATER_COLS_6), new_raters],
        axis=1
    )

    # save
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
