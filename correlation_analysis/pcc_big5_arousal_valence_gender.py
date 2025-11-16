import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ======== configurable area ========
csv_path = 'path/to/iemocap.csv'
personality_cols = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
emotion_cols     = ['valence', 'activation']  # activation = arousal
gender_col       = 'gender'
# ==========================

df = pd.read_csv(csv_path)

# ---- normalize gender values (adjust mapping according to your file if needed) ----
# compatible with M/F, male/female, 0/1 etc.
gender_map = {
    'M':'Male', 'm':'Male', 'male':'Male', 'Male':'Male', 1:'Male', 1.0:'Male',
    'F':'Female', 'f':'Female', 'female':'Female', 'Female':'Female', 0:'Female', 0.0:'Female'
}
df['_gender_norm'] = df[gender_col].map(lambda x: gender_map.get(x, x))

# ---- helper functions for correlations ----
def corr_one_pair(x: pd.Series, y: pd.Series):
    """Remove NaNs and compute Pearson r and p; if n<3, return (nan, nan, n)."""
    tmp = pd.DataFrame({'x': x, 'y': y}).dropna()
    n = len(tmp)
    if n < 3:
        return np.nan, np.nan, n
    r, p = pearsonr(tmp['x'], tmp['y'])
    return r, p, n

def corr_table(sub_df: pd.DataFrame, label: str):
    """Return a long-format DataFrame: group, personality, emotion, r, p, n."""
    rows = []
    for pcol in personality_cols:
        for ecol in emotion_cols:
            r, p, n = corr_one_pair(sub_df[pcol], sub_df[ecol])
            rows.append({
                'group': label,
                'personality': pcol,
                'emotion': ecol,
                'r': r,
                'p': p,
                'n': n
            })
    return pd.DataFrame(rows)

# ---- assemble subgroups: Overall / Male / Female ----
results = []
# Overall
results.append(corr_table(df, 'Overall'))

# Male / Female (only if the gender exists)
for g in ['Male', 'Female']:
    sub = df[df['_gender_norm'] == g]
    if len(sub) > 0:
        results.append(corr_table(sub, g))

res_long = pd.concat(results, ignore_index=True)

with open('pcc_utterance_level_by_gender.txt', 'w') as f:
    for g in res_long['group'].unique():
        sub = res_long[res_long['group'] == g]
        f.write(f'=== {g} ===\n')
        for pcol in personality_cols:
            for ecol in emotion_cols:
                row = sub[(sub['personality'] == pcol) & (sub['emotion'] == ecol)]
                if not row.empty:
                    r = row['r'].values[0]
                    p = row['p'].values[0]
                    n = int(row['n'].values[0])
                    f.write(f'{pcol} vs {ecol}: r={r:.3f}, p={p:.3e}, n={n}\n')
        f.write('\n')

print('Saved: pcc_by_gender.txt')
