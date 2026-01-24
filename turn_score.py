# Data source: https://data.bris.ac.uk/data/dataset/9e748876b7bf30218ef7e4ec4d7f026a
import pandas as pd

df = pd.read_excel("Turning_human_labels.xlsx")

df['turn_type'] = df['number_of_turning_steps'].apply(lambda x: 'pivot' if x <= 2 else 'step')

quantiles = (
    df.groupby(['turn_type', 'turning_angle'])['turning_duration']
    .quantile([0.25, 0.5, 0.75])
    .unstack(level=2)
    .rename(columns={0.25: '25%', 0.5: '50%', 0.75: '75%'})
)

def assign_score(turn_type, angle, duration):
    try:
        q25 = quantiles.loc[(turn_type, angle), '25%']
        q50 = quantiles.loc[(turn_type, angle), '50%']
        q75 = quantiles.loc[(turn_type, angle), '75%']
    except KeyError:
    if duration < q25:
        return 0
    elif duration < q50:
        return 1
    elif duration < q75:
        return 2
    else:
        return 3

df['turning_score'] = df.apply(lambda row: assign_score(row['turn_type'], row['turning_angle'], row['turning_duration']), axis=1)

print(" 分数分布 (pivot vs step):")
for (ttype, angle), group in df.groupby(['turn_type', 'turning_angle']):
    print(f"\n {ttype.upper()} | Angle {angle}")
    print(group['turning_score'].value_counts().sort_index())

df.to_excel("Turning_with_pivot_step_scores.xlsx", index=False)
print("\n 打分完成，保存为 'Turning_with_pivot_step_scores.xlsx'")
