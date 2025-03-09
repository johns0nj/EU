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

# 输出 DataFrame 的前几行
print(df.head())
