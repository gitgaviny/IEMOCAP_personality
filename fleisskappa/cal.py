import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# Load the dataset
file_path = 'ocean_5raters_bin.csv'
df = pd.read_csv(file_path)

# Convert all rater columns to numeric, coercing errors to NaN, then drop rows with any NaN values
rater_columns = ['rater1', 'rater2', 'rater3', 'rater4', 'rater5']
df[rater_columns] = df[rater_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (non-numeric ratings)
df_cleaned = df.dropna(subset=rater_columns)

# Function to calculate Fleiss' Kappa for each tipi question
def calculate_fleiss_kappa(df):
    tipi_groups = df.groupby('tipi')
    kappa_results = {}

    for tipi, group in tipi_groups:
        # Extract ratings from all raters and convert them into a matrix
        ratings_matrix = group[rater_columns].to_numpy()

        # Need to convert the ratings into a binary agreement matrix of counts per category per item
        max_rating = int(ratings_matrix.max())  # Determine maximum rating to create proper category counts
        category_counts = []

        for row in ratings_matrix:
            counts = np.zeros(max_rating)  # Initialize count array for each category
            for rating in row:
                counts[int(rating) - 1] += 1  # Populate counts for each rating (adjusted for zero-indexing)
            category_counts.append(counts)

        # Calculate Fleiss' Kappa for the current tipi question
        kappa = fleiss_kappa(category_counts, method='fleiss')
        kappa_results[tipi] = kappa

    return kappa_results

# Calculate Fleiss' Kappa for each question
kappa_per_tipi = calculate_fleiss_kappa(df_cleaned)

# Save the results to a CSV file
output_file_path = 'fleiss_kappa.csv'
with open(output_file_path, 'w') as f:
    f.write('tipi,Fleiss_Kappa\n')
    for tipi, kappa in kappa_per_tipi.items():
        f.write(f'{tipi},{kappa:.4f}\n')

print(f'Fleiss Kappa results saved to {output_file_path}')
