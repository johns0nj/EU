import pandas as pd

# 读取 RAI.xlsx 文件
df = pd.read_excel('RAI.xlsx')

# 打印列名以检查
print("列名:", df.columns)

# 重命名列
df = df.rename(columns={'World': 'Headline'})

# 确保读取 Momentum 列（注意大小写）
if 'Momentum' not in df.columns:
    raise ValueError("RAI.xlsx 文件中缺少 'Momentum' 列")

# 假设日期列名为'Date'，RAI值列名为'Momentum'
if 'Date' not in df.columns:
    raise ValueError("RAI.xlsx 文件中缺少 'Date' 列")

# 确保日期列为datetime类型
df['Date'] = pd.to_datetime(df['Date'])

# 获取最新一行（日期最大）
latest_row = df.loc[df['Date'].idxmax()]

print(f"最新日期: {latest_row['Date'].strftime('%Y-%m-%d')}")
print(f"最新RAI值: {latest_row['Momentum']}")

# 输出 DataFrame 的前几行
print(df.head())
