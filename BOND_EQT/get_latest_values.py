import pandas as pd

# 读取RAI数据
df = pd.read_excel('RAI.xlsx')

# 打印列名
print("列名:", df.columns.tolist())

# 获取最新的RAI和Momentum值
latest_rai = df['World'].iloc[-1]  # 使用'World'而不是'Headline'
latest_momentum = df['Momentum'].iloc[-1]

# 计算Momentum的变化（过去一个月）
momentum_change = df['Momentum'].diff(21).iloc[-1]  # 21个交易日约等于一个月

# 标准化Momentum变化到[0,1]区间
min_change = df['Momentum'].diff(21).min()
max_change = df['Momentum'].diff(21).max()
normalized_momentum_change = (momentum_change - min_change) / (max_change - min_change)

# 打印结果
print(f"最新RAI值: {latest_rai:.2f}")
print(f"最新Momentum值: {latest_momentum:.2f}")
print(f"Momentum一个月变化值: {momentum_change:.2f}")
print(f"Momentum一个月标准化变化值: {normalized_momentum_change:.2f}") 