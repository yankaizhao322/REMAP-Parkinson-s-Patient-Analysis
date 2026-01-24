# Data source: https://data.bris.ac.uk/data/dataset/9e748876b7bf30218ef7e4ec4d7f026a
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df_updrs = pd.read_excel("SitToStand_human_labels.xls")
df_turning = pd.read_excel("Turning_with_scores.xlsx")

df_updrs = df_updrs.rename(columns={
    'sts_final_attempt_duration': 'final_duration',
    'sts_whole_episode_duration': 'whole_duration',
    'STS_additional_features': 'features',
    'MDS-UPDRS_score_3.9 _arising_from_chair': 'updrs_score'
})

merged = pd.merge(df_updrs, df_turning, on='Participant ID number', how='inner')

if 'On_or_Off_medication' in merged.columns and 'DBS_state' in merged.columns:
    merged['state_combo'] = merged['On_or_Off_medication'].astype(str) + "_" + merged['DBS_state'].astype(str)
else:
    merged['state_combo'] = 'unknown'

merged['features'] = merged['features'].fillna('none')
merged['state_combo'] = merged['state_combo'].fillna('none')

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration', 'turning_duration']),
    ('txt1', CountVectorizer(), 'features'),
    ('txt2', CountVectorizer(), 'state_combo')
])

pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)

pipeline.fit(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']], merged['updrs_score'])
merged['predicted_updrs'] = pipeline.predict(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']])
print(" UPDRS Score 预测报告：")
print(classification_report(merged['updrs_score'], merged['predicted_updrs']))

pipeline.fit(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']], merged['turning_score'])
merged['predicted_turning'] = pipeline.predict(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']])
print("\n Turning Score 预测报告：")
print(classification_report(merged['turning_score'], merged['predicted_turning']))

merged.to_excel("multi_task_predictions.xlsx", index=False)
print("\n 结果保存为 'multi_task_predictions.xlsx'")

wrong_preds = merged[merged['predicted_updrs'] != merged['updrs_score']]
print(wrong_preds[['Participant ID number', 'features', 'updrs_score', 'predicted_updrs']])
