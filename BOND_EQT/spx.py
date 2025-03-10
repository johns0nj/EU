import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html
import numpy as np
from datetime import datetime, timedelta

# 读取数据
df = pd.read_excel('SPXA.xlsx')

# 确保日期列是datetime类型
df['Date'] = pd.to_datetime(df['Date'])

# 筛选1930年以来的数据
df = df[df['Date'] >= '1930-01-01']

# 计算4年移动均线 (约1000个交易日)
df['MA_4Y'] = df['Last Price'].rolling(window=1000).mean()

# 获取最新有效数据
latest_idx = -1
while pd.isna(df['Last Price'].iloc[latest_idx]) and abs(latest_idx) < len(df):
    latest_idx -= 1
latest_date = df['Date'].iloc[latest_idx]
latest_price = df['Last Price'].iloc[latest_idx]

# 创建Dash应用
app = Dash(__name__)

# 创建图表
fig = go.Figure()

# 添加主要的价格线
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['Last Price'],
        mode='lines',
        name='S&P 500',
        line=dict(color='navy', width=1),
        hovertemplate='日期: %{x|%Y-%m-%d}<br>价格: %{y:.2f}<extra></extra>'
    )
)

# 添加4年移动均线
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['MA_4Y'],
        mode='lines',
        name='4年移动均线',
        line=dict(color='red', width=1.5),
        hovertemplate='日期: %{x|%Y-%m-%d}<br>4年均线: %{y:.2f}<extra></extra>'
    )
)

# 添加时间跨度标记 - 从1932年6月开始，每个跨度为16年9个月
start_date = datetime(1932, 6, 1)
span_length = 16 * 12 + 9  # 16年9个月，以月为单位
end_date = datetime(2050, 12, 31)  # 延长结束日期以容纳新的time span

# 计算时间跨度的结束点
current_date = start_date
span_dates = []

while current_date < end_date:
    span_dates.append(current_date)
    # 添加16年9个月
    years_to_add = 16
    months_to_add = 9
    new_year = current_date.year + years_to_add
    new_month = current_date.month + months_to_add
    
    # 处理月份溢出
    if new_month > 12:
        new_year += 1
        new_month -= 12
    
    current_date = datetime(new_year, new_month, 1)

# 添加垂直线和标签来标记每个时间跨度
for i, date in enumerate(span_dates):
    # 添加垂直线
    fig.add_vline(
        x=date,
        line_width=1,
        line_dash="solid",
        line_color="blue"
    )
    
    # 添加标签
    if i < len(span_dates) - 1:
        mid_point = date + (span_dates[i+1] - date) / 2
        fig.add_annotation(
            x=mid_point,
            y=0.02,
            xref="x",
            yref="paper",
            text=f"16年9个月",
            showarrow=False,
            font=dict(size=10, color="blue"),
            align="center",
            bordercolor="blue",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )

# 更新布局，添加十字跟踪光标
fig.update_layout(
    title="S&P 500 (1930年至今) - 对数坐标",
    xaxis=dict(
        title="年份",
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        range=[df['Date'].min(), datetime(2050, 12, 31)]  # 延长x轴以显示新的time span
    ),
    yaxis=dict(
        title="价格",
        type="log",  # 使用对数坐标
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgrey',
        showline=True,
        linewidth=1,
        linecolor='black',
        tickformat=".0f"  # 显示完整数字，不使用科学计数法
    ),
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255, 255, 255, 0.8)'
    ),
    plot_bgcolor='white',
    height=600,
    # 添加十字跟踪光标
    hovermode='x unified',  # 统一x轴上的悬停显示
    hoverdistance=100,      # 悬停检测距离
    spikedistance=1000,     # 尖峰线检测距离
)

# 添加十字跟踪光标的配置
fig.update_xaxes(
    showspikes=True,        # 显示x轴尖峰线
    spikethickness=1,       # 尖峰线粗细
    spikecolor="gray",      # 尖峰线颜色
    spikemode="across",     # 尖峰线模式 - 跨越整个图表
    spikesnap="cursor",     # 尖峰线对齐到最近的数据点
    spikedash="solid"       # 尖峰线样式
)

fig.update_yaxes(
    showspikes=True,        # 显示y轴尖峰线
    spikethickness=1,       # 尖峰线粗细
    spikecolor="gray",      # 尖峰线颜色
    spikemode="across",     # 尖峰线模式 - 跨越整个图表
    spikesnap="cursor",     # 尖峰线对齐到最近的数据点
    spikedash="solid"       # 尖峰线样式
)

# 创建Dash布局
app.layout = html.Div([
    # 添加最新数据显示
    html.Div([
        html.H3(f"最新日期: {latest_date.strftime('%Y-%m-%d')}", 
                style={'margin': '10px', 'textAlign': 'center'}),
        html.H3(f"收盘价: {latest_price:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'color': 'navy'})
    ]),
    dcc.Graph(
        id='spx-log-chart',
        figure=fig,
        config={
            'displayModeBar': True,
            'scrollZoom': True,  # 启用滚轮缩放
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape']  # 添加绘图工具
        }
    )
])

if __name__ == '__main__':
    print("应用已启动，请访问 http://127.0.0.1:8050/")
    app.run_server(debug=True)
