import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.linear_model import LinearRegression

# 读取数据
df = pd.read_excel('reg.xlsx')

# 筛选2020年7月以来的数据
df = df[df.iloc[:, 0] >= '2020-07-01']  # 假设日期在第一列

# 提取X和Y列
X = df.iloc[:, 2].values.reshape(-1, 1)  # C列作为X
Y = df.iloc[:, 1].values  # B列作为Y

# 计算线性回归
reg = LinearRegression()
reg.fit(X, Y)
Y_pred = reg.predict(X)

# 计算残差和标准差
residuals = Y - Y_pred
std_dev = np.std(residuals)

# 获取最新值
latest_X = X[-1][0]
latest_Y = Y[-1]
latest_date = df.iloc[-1, 0]  # 假设日期在第一列

# 获取近6个月的数据
six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
recent_data = df[df.iloc[:, 0] >= six_months_ago]

# 创建Dash应用
app = Dash(__name__)

# 创建图表
fig = go.Figure()

# 添加原始数据点
fig.add_trace(
    go.Scatter(
        x=X.flatten(),
        y=Y,
        mode='markers',
        name='原始数据',
        marker=dict(color='lightcoral', size=8),  # 改为淡红色
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    )
)

# 添加近6个月的数据路径
fig.add_trace(
    go.Scatter(
        x=recent_data.iloc[:, 2],  # C列作为X
        y=recent_data.iloc[:, 1],  # B列作为Y
        mode='lines+markers',
        name='近6个月路径',
        line=dict(color='navy', width=2),  # 改为海军蓝色
        marker=dict(size=8),
        hovertemplate='日期: %{text|%Y-%m-%d}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
        text=recent_data.iloc[:, 0]  # 添加日期信息
    )
)

# 添加回归线
fig.add_trace(
    go.Scatter(
        x=X.flatten(),
        y=Y_pred,
        mode='lines',
        name='回归线',
        line=dict(color='red', width=3),  # 改为红色并保持加粗
        hovertemplate='X: %{x:.2f}<br>预测Y: %{y:.2f}<extra></extra>'
    )
)

# 添加+1倍标准差线
fig.add_trace(
    go.Scatter(
        x=X.flatten(),
        y=Y_pred + std_dev,
        mode='lines',
        name='+1标准差',
        line=dict(color='black', width=2, dash='dot'),  # 改为黑色
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    )
)

# 添加-1倍标准差线
fig.add_trace(
    go.Scatter(
        x=X.flatten(),
        y=Y_pred - std_dev,
        mode='lines',
        name='-1标准差',
        line=dict(color='black', width=2, dash='dot'),  # 改为黑色
        hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    )
)

# 标注最新值
fig.add_trace(
    go.Scatter(
        x=[latest_X],
        y=[latest_Y],
        mode='markers+text',
        name='最新值',
        marker=dict(
            color='green', 
            size=14,  # 从12增大20%到14
            symbol='diamond'  # 改为钻石形状
        ),
        text=[f'最新值<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {latest_X:.2f}<br>Y: {latest_Y:.2f}'],
        textposition="top right",  # 将标注放在点的右上方
        hovertemplate='日期: %{text}<extra></extra>'
    )
)

# 更新布局
fig.update_layout(
    title="恒生科技1年前瞻估值vs中美利差回归分析",
    title_font=dict(size=24),  # 标题字体保持不变
    xaxis=dict(
        title="X值：中美10年期国债收益率利差（%）",
        title_font=dict(size=22),  # 从18增大20%到22
        tickfont=dict(size=17),    # 从14增大20%到17
        showline=True,
        linewidth=2,
        linecolor='black'
    ),
    yaxis=dict(
        title="Y值：恒生科技1年前瞻估值（x）",
        title_font=dict(size=22),  # 从18增大20%到22
        tickfont=dict(size=17),    # 从14增大20%到17
        showline=True,
        linewidth=2,
        linecolor='black'
    ),
    legend=dict(
        font=dict(size=17)  # 从14增大20%到17
    ),
    showlegend=True,
    plot_bgcolor='white',
    height=600,
    width=960  # 设置宽度为960，保持16:10的长宽比
)

# 创建Dash布局
app.layout = html.Div([
    html.Div([
        html.H3(f"回归方程: Y = {reg.coef_[0]:.4f} * X + {reg.intercept_:.4f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),  # 增大方程字体
        html.H3(f"最新值: X = {latest_X:.2f}, Y = {latest_Y:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'})  # 增大最新值字体
    ]),
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
    print("应用已启动，请访问 http://127.0.0.1:8050/")
    app.run_server(debug=True)
