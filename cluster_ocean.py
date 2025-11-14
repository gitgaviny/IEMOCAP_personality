import pandas as pd

# Read data
df = pd.read_csv("path/to/labels.csv")

# Define TIPI mapping
pair_map = {
    (1, 6): "EX",
    (2, 7): "AG",
    (3, 8): "CO",
    (4, 9): "NE",
    (5, 10): "OP",
}

results = []

# Group by dialogue_id + gender (each group is one speaker)
for (dialogue_id, gender), group in df.groupby(["dialogue_id", "gender"]):

    # Use TIPI as index for convenient lookup
    grp = group.set_index("tipi")

    # Prepare one row per speaker
    row = {
        "dialogue_id": dialogue_id,
        "gender": gender,
    }

    for (a, b), trait in pair_map.items():
        if a in grp.index and b in grp.index:

            # Special case: TIPI 2 and 7 -> use (tipi7 - tipi2 + 8) * 0.5
            if (a, b) == (2, 7):
                val = (
                    grp.loc[b, ["rater1", "rater2", "rater3", "rater4", "rater5", "rater6"]]
                    - grp.loc[a, ["rater1", "rater2", "rater3", "rater4", "rater5", "rater6"]]
                    + 8
                ) / 2
            else:
                # Default case: use (a - b + 8) * 0.5
                val = (
                    grp.loc[a, ["rater1", "rater2", "rater3", "rater4", "rater5", "rater6"]]
                    - grp.loc[b, ["rater1", "rater2", "rater3", "rater4", "rater5", "rater6"]]
                    + 8
                ) / 2

            # Mean of rater1–rater6, rounded to 3 decimals
            averaged = round(val.mean(), 3)

            # Save this trait's averaged value into its own column
            row[trait] = averaged

    results.append(row)

# Create new DataFrame: one row per speaker, 5 TIPI columns
out_df = pd.DataFrame(results)

# Optional: enforce column order
cols = ["dialogue_id", "gender", "EX", "AG", "CO", "NE", "OP"]
out_df = out_df[cols]

# Save
out_df.to_csv("ocean.csv", index=False)

