import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.linear_model import LinearRegression

# 读取Excel文件中的所有数据
df = pd.read_excel('intreg.xlsx')

# 打印整个DataFrame
print("读取的数据：")
print(df)

# 打印DataFrame的基本信息
print("\n数据信息：")
print(df.info())

# 打印描述性统计（仅对数值列）
print("\n描述性统计：")
print(df.describe())

# 打印前5行数据
print("\n前5行数据：")
print(df.head())

# 打印后5行数据
print("\n后5行数据：")
print(df.tail())

# 计算市值占比
df['市值占比'] = df.groupby('地区')['市值'].transform(lambda x: x / x.sum())

# 分割数据
us_data = df[df['地区'] == '美国']
cn_data = df[df['地区'] == '中国']

# 定义回归分析函数
def weighted_regression(data):
    X = data['未来3年收入CAGR'].values.reshape(-1, 1)
    y = data['前瞻PS'].values
    weights = data['市值占比'].values
    
    reg = LinearRegression()
    reg.fit(X, y, sample_weight=weights)
    y_pred = reg.predict(X)
    
    return X, y, y_pred, reg

# 进行回归分析
us_X, us_y, us_y_pred, us_reg = weighted_regression(us_data)
cn_X, cn_y, cn_y_pred, cn_reg = weighted_regression(cn_data)

# 创建Dash应用
app = Dash(__name__)

# 创建图表
fig = go.Figure()

# 添加美国数据
fig.add_trace(
    go.Scatter(
        x=us_X.flatten(),
        y=us_y,
        mode='markers+text',
        name='美国数据',
        marker=dict(color='blue', size=10),
        text=us_data['简称'],
        textposition='top center',
        textfont=dict(color='darkblue'),
        hovertemplate='简称: %{text}<br>市值占比: %{customdata:.2%}<br>未来3年收入CAGR: %{x:.2f}%<br>前瞻PS: %{y:.2f}<extra></extra>',
        customdata=us_data['市值占比']
    )
)

# 添加美国回归线
fig.add_trace(
    go.Scatter(
        x=us_X.flatten(),
        y=us_y_pred,
        mode='lines',
        name='美国回归线',
        line=dict(color='blue', width=2),
        hovertemplate='未来3年收入CAGR: %{x:.2f}%<br>预测前瞻PS估值: %{y:.2f}<extra></extra>'
    )
)

# 添加中国数据
fig.add_trace(
    go.Scatter(
        x=cn_X.flatten(),
        y=cn_y,
        mode='markers+text',
        name='中国数据',
        marker=dict(color='red', size=10),
        text=cn_data['简称'],
        textposition='top center',
        textfont=dict(color='darkred'),
        hovertemplate='简称: %{text}<br>市值占比: %{customdata:.2%}<br>未来3年收入CAGR: %{x:.2f}%<br>前瞻PS: %{y:.2f}<extra></extra>',
        customdata=cn_data['市值占比']
    )
)

# 添加中国回归线
fig.add_trace(
    go.Scatter(
        x=cn_X.flatten(),
        y=cn_y_pred,
        mode='lines',
        name='中国回归线',
        line=dict(color='red', width=2),
        hovertemplate='未来3年收入CAGR: %{x:.2f}%<br>预测前瞻PS估值: %{y:.2f}<extra></extra>'
    )
)

# 更新布局
fig.update_layout(
    title="中美企业未来3年收入CAGR与前瞻PS回归分析",
    title_font=dict(size=24),
    xaxis=dict(
        title="未来3年收入CAGR (%)",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        tickformat=".2%",
        showline=True,
        linewidth=2,
        linecolor='black'
    ),
    yaxis=dict(
        title="前瞻PS",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showline=True,
        linewidth=2,
        linecolor='black'
    ),
    legend=dict(
        font=dict(size=14)
    ),
    showlegend=True,
    plot_bgcolor='white',
    height=600,
    width=960
)

# 创建Dash布局
app.layout = html.Div([
    html.H3("美国回归方程: Y = {:.4f} * X + {:.4f}".format(us_reg.coef_[0], us_reg.intercept_)),
    html.H3("中国回归方程: Y = {:.4f} * X + {:.4f}".format(cn_reg.coef_[0], cn_reg.intercept_)),
    dcc.Graph(
        id='regression-chart',
        figure=fig,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
