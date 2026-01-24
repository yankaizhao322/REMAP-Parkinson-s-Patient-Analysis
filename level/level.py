import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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

for col in ['On_or_Off_medication', 'DBS_state']:
    train_df[col] = train_df[col].replace('-', 'unknown')
    test_df[col] = test_df[col].replace('-', 'unknown')

train_df['features'] = train_df['features'].fillna('none')
test_df['features'] = test_df['features'].fillna('none')

def simplify(score):
    if score in [0, 1]:
        return 0  # mild
    elif score == 2:
        return 1  # moderate
    else:
        return 2  # severe

train_df['true_score'] = train_df['true_score'].apply(simplify)
test_df['true_score'] = test_df['true_score'].apply(simplify)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration']),
    ('txt', CountVectorizer(), 'features'),
    ('med', OneHotEncoder(handle_unknown='ignore'), ['On_or_Off_medication']),
    ('dbs', OneHotEncoder(handle_unknown='ignore'), ['DBS_state']),
])

pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42)
)

pipeline.fit(train_df[['final_duration', 'whole_duration', 'features',
                       'On_or_Off_medication', 'DBS_state']],
             train_df['true_score'])

y_pred = pipeline.predict(test_df[['final_duration', 'whole_duration', 'features',
                                   'On_or_Off_medication', 'DBS_state']])

accuracy = accuracy_score(test_df['true_score'], y_pred)
report = classification_report(test_df['true_score'], y_pred)

print(" Accuracy:", round(accuracy * 100, 2), "%")
print(" Classification Report:\n", report)

test_df['Predicted'] = y_pred
test_df['Correct'] = test_df['Predicted'] == test_df['true_score']
wrong = test_df[test_df['Correct'] == False][[
    'Transition ID', 'Participant ID number', 'features', 'final_duration',
    'On_or_Off_medication', 'DBS_state', 'true_score', 'Predicted'
]]

print("\n Misclassified Samples:")
print(wrong.to_string(index=False))
