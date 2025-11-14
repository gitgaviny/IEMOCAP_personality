import pandas as pd
from scipy.stats import pearsonr

# === 1) Load data ===
CSV_PATH = 'path/to/iemocap.csv'  # Change to your path
df = pd.read_csv(CSV_PATH)

# === 2) Column names ===
personality_cols = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
emotion_cols = ['valence', 'activation']

# === 3) Aggregate emotion scores per speaker (mean) ===
agg_emotion = (
    df.groupby('speaker', as_index=False)[emotion_cols]
      .mean()
      .rename(columns={'valence': 'valence_mean', 'activation': 'activation_mean'})
)

# === 4) Extract personality for each speaker 
# (assuming personality is constant per speaker, take first occurrence) ===
personality_by_speaker = (
    df.sort_values('speaker')
      .groupby('speaker', as_index=False)[personality_cols]
      .first()
)

# === 5) Merge personality + mean emotion for each speaker ===
merged = pd.merge(personality_by_speaker, agg_emotion, on='speaker', how='inner')

# Drop speakers with missing values if any
merged = merged.dropna(subset=personality_cols + ['valence_mean', 'activation_mean'])

# === 6) Compute Pearson correlation ===
pairs = [(p, 'valence_mean') for p in personality_cols] + [(p, 'activation_mean') for p in personality_cols]
pcc_results = {}

for p, e in pairs:
    r, pval = pearsonr(merged[p], merged[e])
    pcc_results[(p, e)] = (r, pval)

# === 7) Save and print results ===
OUTPUT = 'pcc_results_conversational_level.txt'
with open(OUTPUT, 'w', encoding='utf-8') as f:
    for (p, e), (r, pval) in pcc_results.items():
        line = f'PCC between {p} and {e}: {r:.3f}, p-value: {pval:.3e}\n'
        f.write(line)
        print(line, end='')

print(f'\nResults written to: {OUTPUT}')
