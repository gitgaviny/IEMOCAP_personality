import pandas as pd
import pingouin as pg


data = pd.read_csv('path/to/ocean_5raters.csv')

icc_results = {}

for task in data['tipi'].unique():
    task_data = data[data['tipi'] == task]
    ratings = task_data[['rater1', 'rater2', 'rater3', 'rater4', 'rater5']]

    ratings_long = ratings.melt(ignore_index=False).reset_index()
    ratings_long.columns = ['sample', 'rater', 'rating']

    icc = pg.intraclass_corr(data=ratings_long, targets='sample', raters='rater', ratings='rating')
    icc_results[task] = icc

icc_summary = pd.concat(icc_results, names=['task']).reset_index(level=0)
print(icc_summary)