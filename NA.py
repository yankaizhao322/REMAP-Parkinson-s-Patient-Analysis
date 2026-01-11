# 100%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_excel("NA.xlsx")
test_df = pd.read_excel("test_NA.xlsx")

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

# Feature preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration']),
    ('txt', CountVectorizer(), 'features')
])

# Build model pipeline
pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)

pipeline.fit(train_df[['final_duration', 'whole_duration', 'features']], train_df['true_score'])

y_pred = pipeline.predict(test_df[['final_duration', 'whole_duration', 'features']])

accuracy = accuracy_score(test_df['true_score'], y_pred)
report = classification_report(test_df['true_score'], y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
print("Classification Report:\n", report)

test_df['Predicted UPDRS_3.9'] = y_pred
test_df['Correct Prediction'] = test_df['Predicted UPDRS_3.9'] == test_df['true_score']

results_df = test_df[[
    'Transition ID',
    'Participant ID number',
    'true_score',
    'Predicted UPDRS_3.9',
    'Correct Prediction'
]].rename(columns={'true_score': 'Actual UPDRS_3.9'})

print(results_df.to_string(index=False))

vectorizer = CountVectorizer()
vectorizer.fit(train_df['features'])

print(vectorizer.get_feature_names_out())
