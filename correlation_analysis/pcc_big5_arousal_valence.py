import pandas as pd
from scipy.stats import pearsonr

# Load the uploaded CSV file
data = pd.read_csv('path/to/iemocap.csv')

# Select the personality traits and emotion percentage columns
personality_columns = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
emotion_percentage_columns = ['valence', 'activation']

# Initialize a dictionary to store the Pearson correlation coefficients (PCC) and p-values
pcc_results = {}

# Calculate the PCC and p-values between each personality trait and each emotion percentage
for personality in personality_columns:
    for emotion in emotion_percentage_columns:
        # Calculate the Pearson correlation coefficient and p-value
        correlation, p_value = pearsonr(data[personality], data[emotion])
        # Store the results in the dictionary
        pcc_results[(personality, emotion)] = (correlation, p_value)

# Write the calculated PCC values and p-values to a file
with open('pcc_results_utterance_level.txt', 'w') as file:
    for (personality, emotion), (pcc, p_value) in pcc_results.items():
        file.write(f'PCC between {personality} and {emotion}: {pcc:.3f}, p-value: {p_value:.3e}\n')

# Print the calculated PCC values and p-values
for (personality, emotion), (pcc, p_value) in pcc_results.items():
    print(f'PCC between {personality} and {emotion}: {pcc:.3f}, p-value: {p_value:.3e}')
