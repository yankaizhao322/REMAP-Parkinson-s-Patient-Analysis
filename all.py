import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === 1. 读取数据 ===
df_updrs = pd.read_excel("SitToStand_human_labels.xls")
df_turning = pd.read_excel("Turning_with_scores.xlsx")

# === 2. 重命名列，确保一致性 ===
df_updrs = df_updrs.rename(columns={
    'sts_final_attempt_duration': 'final_duration',
    'sts_whole_episode_duration': 'whole_duration',
    'STS_additional_features': 'features',
    'MDS-UPDRS_score_3.9 _arising_from_chair': 'updrs_score'
})

# === 3. 合并两个文件（按 Participant ID）===
merged = pd.merge(df_updrs, df_turning, on='Participant ID number', how='inner')

# === 4. 检查状态列是否存在 ===
if 'On_or_Off_medication' in merged.columns and 'DBS_state' in merged.columns:
    merged['state_combo'] = merged['On_or_Off_medication'].astype(str) + "_" + merged['DBS_state'].astype(str)
else:
    merged['state_combo'] = 'unknown'

# === 5. 填补空值 ===
merged['features'] = merged['features'].fillna('none')
merged['state_combo'] = merged['state_combo'].fillna('none')

# === 6. 预处理：数值 + 文本 ===
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration', 'turning_duration']),
    ('txt1', CountVectorizer(), 'features'),
    ('txt2', CountVectorizer(), 'state_combo')
])

# === 7. Pipeline 模型（你可以换成 transformer later）===
pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)

# === 8. Task 1: 预测站起评分（UPDRS 3.9）===
pipeline.fit(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']], merged['updrs_score'])
merged['predicted_updrs'] = pipeline.predict(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']])
print("🎯 UPDRS Score 预测报告：")
print(classification_report(merged['updrs_score'], merged['predicted_updrs']))

# === 9. Task 2: 预测转身评分（Turning Score）===
pipeline.fit(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']], merged['turning_score'])
merged['predicted_turning'] = pipeline.predict(merged[['final_duration', 'whole_duration', 'turning_duration', 'features', 'state_combo']])
print("\n🔁 Turning Score 预测报告：")
print(classification_report(merged['turning_score'], merged['predicted_turning']))

# === 10. 保存结果（可选）===
merged.to_excel("multi_task_predictions.xlsx", index=False)
print("\n📁 结果保存为 'multi_task_predictions.xlsx'")

wrong_preds = merged[merged['predicted_updrs'] != merged['updrs_score']]
print(wrong_preds[['Participant ID number', 'features', 'updrs_score', 'predicted_updrs']])
