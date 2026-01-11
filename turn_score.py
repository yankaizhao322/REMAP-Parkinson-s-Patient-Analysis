import pandas as pd

# === 1. 读取数据 ===
df = pd.read_excel("Turning_human_labels.xlsx")

# === 2. 判断是 Pivot 还是 Step Turn ===
df['turn_type'] = df['number_of_turning_steps'].apply(lambda x: 'pivot' if x <= 2 else 'step')

# === 3. 计算每个 group (turn_type + angle) 的分位数 ===
quantiles = (
    df.groupby(['turn_type', 'turning_angle'])['turning_duration']
    .quantile([0.25, 0.5, 0.75])
    .unstack(level=2)
    .rename(columns={0.25: '25%', 0.5: '50%', 0.75: '75%'})
)

# === 4. 打分函数（按 pivot/step + angle）===
def assign_score(turn_type, angle, duration):
    try:
        q25 = quantiles.loc[(turn_type, angle), '25%']
        q50 = quantiles.loc[(turn_type, angle), '50%']
        q75 = quantiles.loc[(turn_type, angle), '75%']
    except KeyError:
        return -1  # 数据不足的类别标为 -1
    if duration < q25:
        return 0
    elif duration < q50:
        return 1
    elif duration < q75:
        return 2
    else:
        return 3

df['turning_score'] = df.apply(lambda row: assign_score(row['turn_type'], row['turning_angle'], row['turning_duration']), axis=1)

# === 5. 打印每类分布 ===
print("🎯 分数分布 (pivot vs step):")
for (ttype, angle), group in df.groupby(['turn_type', 'turning_angle']):
    print(f"\n🌀 {ttype.upper()} | Angle {angle}")
    print(group['turning_score'].value_counts().sort_index())

# === 6. 保存结果 ===
df.to_excel("Turning_with_pivot_step_scores.xlsx", index=False)
print("\n✅ 打分完成，保存为 'Turning_with_pivot_step_scores.xlsx'")
