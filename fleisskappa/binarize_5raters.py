#!/usr/bin/env python3
import pandas as pd

OCEAN_CSV = "/lustre/users/gao/IEMOCAP_personality/dataset/ocean.csv"  # First file: contains EX, AG, CO, NE, OP
RATERS_CSV_IN = "ocean_5raters.csv"   # Second file: contains tipi, rater1-5, etc.
RATERS_CSV_OUT = "ocean_5raters_bin.csv"   # Output file

# Columns for the five personality traits
TRAIT_COLS = ["EX", "AG", "CO", "NE", "OP"]
# Rater columns
RATER_COLS = [f"rater{i}" for i in range(1, 6)]

def main():
    # 1) Load the first file and compute medians for the 5 traits
    df_ocean = pd.read_csv(OCEAN_CSV)

    # Ensure these trait columns exist
    for c in TRAIT_COLS:
        if c not in df_ocean.columns:
            raise ValueError(f"Missing column in {OCEAN_CSV}: {c}")

    medians = df_ocean[TRAIT_COLS].median().to_dict()
    print("Medians of the five traits:", medians)

    # 2) Load the second file
    df_raters = pd.read_csv(RATERS_CSV_IN)

    # Check required columns
    if "tipi" not in df_raters.columns:
        raise ValueError(f"Missing column in {RATERS_CSV_IN}: tipi")

    for c in RATER_COLS:
        if c not in df_raters.columns:
            raise ValueError(f"Missing column in {RATERS_CSV_IN}: {c}")

    # 3) For each row, use tipi to apply median threshold and convert rater1-5 to 1/2
    def binarize_row(row):
        trait = row["tipi"]  # e.g., "EX", "AG", ...
        if trait not in medians:
            raise ValueError(f"Unknown tipi value: {trait}")

        threshold = medians[trait]

        for col in RATER_COLS:
            val = float(row[col])
            row[col] = 1 if val > threshold else 2
        return row

    df_raters = df_raters.apply(binarize_row, axis=1)

    # 4) Save the output
    df_raters.to_csv(RATERS_CSV_OUT, index=False)
    print(f"Saved to: {RATERS_CSV_OUT}")

if __name__ == "__main__":
    main()
