# Data source: https://data.bris.ac.uk/data/dataset/9e748876b7bf30218ef7e4ec4d7f026a
#100%

# 92.31% 
#15 596 >1 attempt\nmoves forward in chair\nCarrying something           1.618       Off medication   unknown           1          0
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#df_mild = pd.read_excel("mild.xlsx")
#df_moderate = pd.read_excel("moderate.xlsx")
#df_severe = pd.read_excel("severe.xlsx")
#train_df = pd.concat([df_mild, df_moderate, df_severe], ignore_index=True)
train_df = pd.read_excel("On_medication.xlsx")
test_df = pd.read_excel("test_on_medication.xlsx")

rename_map = {
    'sts_final_attempt_duration': 'final_duration',
    'sts_whole_episode_duration': 'whole_duration',
    'STS_additional_features': 'features',
    'MDS-UPDRS_score_3.9 _arising_from_chair': 'true_score'
}
train_df.rename(columns=rename_map, inplace=True)
test_df.rename(columns=rename_map, inplace=True)

for col in ['On_or_Off_medication', 'DBS_state']:
    train_df[col] = train_df[col].replace('-', 'unknown')
    test_df[col] = test_df[col].replace('-', 'unknown')

train_df['features'] = train_df['features'].fillna('none')
test_df['features'] = test_df['features'].fillna('none')

train_df['state_combo'] = train_df['On_or_Off_medication'] + "_" + train_df['DBS_state']
test_df['state_combo'] = test_df['On_or_Off_medication'] + "_" + test_df['DBS_state']

keywords = ['slow', 'arms', 'attempt']
for kw in keywords:
    train_df[f'has_{kw}'] = train_df['features'].str.lower().apply(lambda x: int(kw in x))
    test_df[f'has_{kw}'] = test_df['features'].str.lower().apply(lambda x: int(kw in x))

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), ['final_duration', 'whole_duration', 'has_slow', 'has_arms', 'has_attempt']),
    ('feat_text', CountVectorizer(), 'features'),
    ('state_text', CountVectorizer(), 'state_combo')
])

pipeline = make_pipeline(
    preprocessor,
    RandomForestClassifier(random_state=42, class_weight='balanced')
)

pipeline.fit(train_df[['final_duration', 'whole_duration', 'features', 'state_combo',
                       'has_slow', 'has_arms', 'has_attempt']], train_df['true_score'])

y_pred = pipeline.predict(test_df[['final_duration', 'whole_duration', 'features', 'state_combo',
                                   'has_slow', 'has_arms', 'has_attempt']])

accuracy = accuracy_score(test_df['true_score'], y_pred)
report = classification_report(test_df['true_score'], y_pred)

print(" Accuracy:", round(accuracy * 100, 2), "%")
print(" Classification Report:\n", report)


test_df['Predicted'] = y_pred
test_df['Correct'] = test_df['Predicted'] == test_df['true_score']
wrong = test_df[~test_df['Correct']][[
    'Transition ID', 'features', 'final_duration',
    'On_or_Off_medication', 'DBS_state', 'true_score', 'Predicted'
]]

print("\n Misclassified Samples:")
print(wrong.to_string(index=False))
vocab = pipeline.named_steps['columntransformer'].named_transformers_['feat_text'].get_feature_names_out()
print(vocab)

'''
def simplify(score):
    if score in [0, 1]:
        return 1  # Mild
    elif score == 2:
        return 2  # Moderate
    else:
        return 3  # Severe
train_df['true_score'] = train_df['true_score'].apply(simplify)
test_df['true_score'] = test_df['true_score'].apply(simplify)
'''
