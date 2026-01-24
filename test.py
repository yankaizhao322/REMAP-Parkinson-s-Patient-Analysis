# Data source: https://data.bris.ac.uk/data/dataset/9e748876b7bf30218ef7e4ec4d7f026a
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

train_df['features'] = train_df['features'].fillna('none')
test_df['features'] = test_df['features'].fillna('none')

train_df['state_combo'] = train_df['On_or_Off_medication'].astype(str) + "_" + train_df['DBS_state'].astype(str)
test_df['state_combo'] = test_df['On_or_Off_medication'].astype(str) + "_" + test_df['DBS_state'].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration']),
    ('txt1', CountVectorizer(), 'features'),
])

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

pipeline.fit(train_df[['final_duration', 'whole_duration', 'features', 'state_combo']], train_df['true_score'])

y_pred = pipeline.predict(test_df[['final_duration', 'whole_duration', 'features', 'state_combo']])

accuracy = accuracy_score(test_df['true_score'], y_pred)
report = classification_report(test_df['true_score'], y_pred)

print(" Accuracy:", round(accuracy * 100, 2), "%")
print(" Classification Report:\n", report)

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

print(" Prediction Results:")
print(results_df.to_string(index=False))