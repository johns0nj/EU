import pandas as pd
from datetime import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go

# 读取 SPX.xlsx 文件
df = pd.read_excel('SPX.xlsx')

# 打印列名以检查
print("列名:", df.columns)

# 确保日期列是 datetime 类型，并精确到日
df['Date'] = pd.to_datetime(df['Date']).dt.date

# 设置日期为索引
df.set_index('Date', inplace=True)

# 创建从 2010-01-01 到 2025-03-06 的完整日期范围
full_date_range = pd.date_range(start='2010-01-01', end='2025-03-06', freq='D')

# 重新索引 DataFrame 以包含完整的日期范围
df = df.reindex(full_date_range)

# 使用前向填充和后向填充来补足缺失的收盘数据
# 假设收盘数据列名为 'Last Price'，如果不是，请根据实际列名修改
df['Last Price'].ffill(inplace=True)
df['Last Price'].bfill(inplace=True)

# 重置索引，将日期重新变为一列
df.reset_index(inplace=True)
df.rename(columns={'index': 'Date'}, inplace=True)

# 输出处理后的 DataFrame
print(df)

# 将结果保存到 SPX_processed.xlsx 文件中
df.to_excel('SPX_processed.xlsx', index=False)

# 创建一个简单的图表
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])

# 将图表保存为HTML文件
fig.write_html("output.html")

# Dash应用
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
