import pandas as pd

# ====== paths: change to your own if needed ======
INPUT_CSV = "tipi_labels.csv"   # original file
OUTPUT_CSV = "ocean_6raters.csv" # output file

# Read data
df = pd.read_csv(INPUT_CSV)

# Make sure tipi is integer
df["tipi"] = df["tipi"].astype(int)

# TIPI pair mapping: (tipi_a, tipi_b) -> new trait name
pair_map = {
    (1, 6): "EX",
    (2, 7): "AG",
    (3, 8): "CO",
    (4, 9): "NE",
    (5, 10): "OP",
}

rater_cols = [f"rater{i}" for i in range(1, 7)]
results = []

# Group by speaker: dialogue_id + gender
for (dialogue_id, gender), group in df.groupby(["dialogue_id", "gender"]):
    # Use tipi as index so we can look up by tipi value
    grp = group.set_index("tipi")

    for (a, b), trait in pair_map.items():
        # Strict check: all tipi must exist
        if a not in grp.index or b not in grp.index:
            raise ValueError(
                f"Missing tipi {a} or {b} for dialogue_id={dialogue_id}, gender={gender}"
            )

        row_a = grp.loc[a]
        row_b = grp.loc[b]

        new_row = {
            "dialogue_id": dialogue_id,
            "gender": gender,
            "tipi": trait,  # EX / AG / CO / NE / OP
        }

        # Process each rater separately
        for col in rater_cols:
            if (a, b) == (2, 7):
                # Special case: use (tipi7 - tipi2 + 8) * 0.5
                new_row[col] = (row_b[col] - row_a[col] + 8) * 0.5
            else:
                # Default: (tipi_a - tipi_b + 8) * 0.5
                new_row[col] = (row_a[col] - row_b[col] + 8) * 0.5

        results.append(new_row)

# Build new DataFrame
out_df = pd.DataFrame(results)

# ====== sort by desired tipi order: OP, CO, EX, AG, NE ======
trait_order = ["OP", "CO", "EX", "AG", "NE"]
out_df["tipi"] = pd.Categorical(out_df["tipi"], categories=trait_order, ordered=True)
out_df = out_df.sort_values(["dialogue_id", "gender", "tipi"])

# Save
out_df.to_csv(OUTPUT_CSV, index=False)

print("Done! Saved to:", OUTPUT_CSV)
