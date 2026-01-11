#92.98%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

df_mild = pd.read_excel("mild.xlsx")
df_moderate = pd.read_excel("moderate.xlsx")
df_severe = pd.read_excel("severe.xlsx")
train_df = pd.concat([df_mild, df_moderate, df_severe], ignore_index=True)

test_df = pd.read_excel("test.xlsx")
rename_map = {
    'sts_final_attempt_duration': 'final_duration',
    'sts_whole_episode_duration': 'whole_duration',
    'STS_additional_features': 'features',
    'MDS-UPDRS_score_3.9 _arising_from_chair': 'true_score'
}
train_df = train_df.rename(columns=rename_map)
test_df = test_df.rename(columns=rename_map)

# === 填补缺失值 ===
train_df['features'] = train_df['features'].fillna('none')
test_df['features'] = test_df['features'].fillna('none')

# === 添加 medication & DBS 状态作为组合特征 ===
train_df['state_combo'] = train_df['On_or_Off_medication'].astype(str) + "_" + train_df['DBS_state'].astype(str)
test_df['state_combo'] = test_df['On_or_Off_medication'].astype(str) + "_" + test_df['DBS_state'].astype(str)

# === 预处理器（包含数值 + 文本 + 状态特征）===
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration']),
    ('txt1', CountVectorizer(), 'features'),
    ('txt2', CountVectorizer(), 'state_combo')  # 加入状态
])

# === 模型管道 ===
pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [10, 20, None]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(train_df[['final_duration', 'whole_duration', 'features', 'state_combo']], train_df['true_score'])

print("Best parameters:", grid.best_params_)
print(train_df['true_score'].value_counts())

# === 模型训练 ===
pipeline.fit(train_df[['final_duration', 'whole_duration', 'features', 'state_combo']], train_df['true_score'])

# === 模型预测 ===
y_pred = pipeline.predict(test_df[['final_duration', 'whole_duration', 'features', 'state_combo']])

# === 结果评估 ===
accuracy = accuracy_score(test_df['true_score'], y_pred)
report = classification_report(test_df['true_score'], y_pred)

print("✅ Accuracy:", round(accuracy * 100, 2), "%")
print("📊 Classification Report:\n", report)

# === 错误预测记录 ===
test_df['Predicted UPDRS_3.9'] = y_pred
test_df['Correct Prediction'] = test_df['Predicted UPDRS_3.9'] == test_df['true_score']

results_df = test_df[[
    'Transition ID',
    'Participant ID number',
    'On_or_Off_medication',
    'DBS_state',
    'features',
    'true_score',
    'Predicted UPDRS_3.9',
    'Correct Prediction'
]].rename(columns={'true_score': 'Actual UPDRS_3.9'})

print("🔍 Prediction Results:")
print(results_df.to_string(index=False))