import pandas as pd
import pingouin as pg

# 读取你的数据
data = pd.read_csv('path/to/ocean_with_filtered_and_binarized.csv')

# 选择要计算的任务
icc_results = {}

for task in data['tipi'].unique():
    task_data = data[data['tipi'] == task]
    ratings = task_data[['filtered_rater1', 'filtered_rater2', 'filtered_rater3', 'filtered_rater4', 'filtered_rater5']]

    # 转换数据为长格式
    ratings_long = ratings.melt(ignore_index=False).reset_index()
    ratings_long.columns = ['sample', 'rater', 'rating']

    # 计算ICC
    icc = pg.intraclass_corr(data=ratings_long, targets='sample', raters='rater', ratings='rating')
    icc_results[task] = icc

# 汇总并显示ICC结果
icc_summary = pd.concat(icc_results, names=['task']).reset_index(level=0)
print(icc_summary)