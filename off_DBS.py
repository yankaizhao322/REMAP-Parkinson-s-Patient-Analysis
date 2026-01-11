import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === 1. 读取数据 ===
train_df = pd.read_excel("off_DBS.xlsx")
test_df = pd.read_excel("test_off_DBS.xlsx")

# === 2. 重命名列 ===
rename_map = {
    'sts_final_attempt_duration': 'final_duration',
    'sts_whole_episode_duration': 'whole_duration',
    'STS_additional_features': 'features',
    'MDS-UPDRS_score_3.9 _arising_from_chair': 'true_score'
}
train_df = train_df.rename(columns=rename_map)
test_df = test_df.rename(columns=rename_map)

# === 3. 替换缺失状态 "-" 为 "unknown" ===
for col in ['On_or_Off_medication', 'DBS_state']:
    train_df[col] = train_df[col].replace('-', 'unknown')
    test_df[col] = test_df[col].replace('-', 'unknown')

# === 4. 合成状态组合特征 ===
train_df['state_combo'] = train_df['On_or_Off_medication'] + "_" + train_df['DBS_state']
test_df['state_combo'] = test_df['On_or_Off_medication'] + "_" + test_df['DBS_state']

# === 5. 填补文本空值 ===
train_df['features'] = train_df['features'].fillna('none')
test_df['features'] = test_df['features'].fillna('none')

# === 6. 构建预处理器（数值 + 特征文本 + 状态文本）===
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration']),
    ('txt1', CountVectorizer(), 'features'),
    ('txt2', CountVectorizer(), 'state_combo')
])

# === 7. 建模管道 ===
pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)

# === 8. 模型训练 ===
pipeline.fit(train_df[['final_duration', 'whole_duration', 'features', 'state_combo']], train_df['true_score'])

# === 9. 预测 + 评估 ===
y_pred = pipeline.predict(test_df[['final_duration', 'whole_duration', 'features', 'state_combo']])
accuracy = accuracy_score(test_df['true_score'], y_pred)
report = classification_report(test_df['true_score'], y_pred)

print("✅ Accuracy:", round(accuracy * 100, 2), "%")
print("📊 Classification Report:\n", report)

# === 10. 附加输出：错误案例 ===
test_df['Predicted UPDRS_3.9'] = y_pred
test_df['Correct'] = test_df['Predicted UPDRS_3.9'] == test_df['true_score']

wrong = test_df[test_df['Correct'] == False][[
    'Transition ID', 'Participant ID number', 'features', 'final_duration',
    'On_or_Off_medication', 'DBS_state', 'true_score', 'Predicted UPDRS_3.9'
]]

print("\n🔍 Misclassified Samples:")
print(wrong.to_string(index=False))
