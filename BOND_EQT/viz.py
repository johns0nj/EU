import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table
from rai import df as rai_df  # 从 rai.py 中导入 RAI 数据
from data_processing import df as spx_df  # 从 data_processing.py 中导入标普500数据
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# 将 RAI 的 Date 列转换为字符串类型
rai_df['Date'] = rai_df['Date'].astype(str)

# 清理 RAI 的 Date 列，只保留有效的日期字符串
rai_df = rai_df[rai_df['Date'].str.match(r'\d{4}-\d{2}-\d{2}')]

# 确保 RAI 的日期列是 datetime 类型
rai_df['Date'] = pd.to_datetime(rai_df['Date'])

# 确保标普500的日期列是 datetime 类型
spx_df['Date'] = pd.to_datetime(spx_df['Date'])

# 计算标普500的未来3个月跌幅
spx_df['Future_3M_Price'] = spx_df['Last Price'].shift(-63)  # 约3个月的交易日
spx_df['Future_3M_Return'] = (spx_df['Future_3M_Price'] - spx_df['Last Price']) / spx_df['Last Price'] * 100

# 找出跌幅超过10%的日期
drop_dates = spx_df[spx_df['Future_3M_Return'] < -10]['Date']

# 获取 RAI 和标普500的日期范围
min_date = max(rai_df['Date'].min(), spx_df['Date'].min())
max_date = min(rai_df['Date'].max(), spx_df['Date'].max())

# 过滤数据，确保日期范围一致
rai_df = rai_df[(rai_df['Date'] >= min_date) & (rai_df['Date'] <= max_date)]
spx_df = spx_df[(spx_df['Date'] >= min_date) & (spx_df['Date'] <= max_date)]

# 计算不同持有期的未来回报
spx_df['3M_Future_Return'] = (spx_df['Last Price'].shift(-63) - spx_df['Last Price']) / spx_df['Last Price'] * 100
spx_df['6M_Future_Return'] = (spx_df['Last Price'].shift(-126) - spx_df['Last Price']) / spx_df['Last Price'] * 100
spx_df['12M_Future_Return'] = (spx_df['Last Price'].shift(-252) - spx_df['Last Price']) / spx_df['Last Price'] * 100

# 检查 RAI 数据中低于 -2 的记录
print("RAI < -2 记录数:", rai_df[rai_df['Headline'] < -2].shape[0])
print("RAI < -1 记录数:", rai_df[rai_df['Headline'] < -1].shape[0])
print("RAI < 0.5 记录数:", rai_df[rai_df['Headline'] < 0.5].shape[0])

# 检查回报率数据
print(spx_df[['Date', 'Last Price', '3M_Future_Return', '6M_Future_Return', '12M_Future_Return']].head(10))

# 检查日期范围
print(f"Min Date: {min_date}, Max Date: {max_date}")

def calculate_win_rate_and_ratio(rai_threshold, return_column, return_threshold):
    # 合并两个 DataFrame，确保日期对齐
    merged_df = pd.merge(spx_df, rai_df, on='Date', how='inner')
    
    # 过滤 RAI 小于阈值的记录
    filtered_df = merged_df[merged_df['Headline'] < rai_threshold]
    
    # 计算胜率和盈亏比
    if len(filtered_df) > 0 and return_column in filtered_df.columns:
        # 计算回报率大于阈值的记录数
        win_count = len(filtered_df[filtered_df[return_column] > return_threshold])
        # 计算胜率
        win_rate = win_count / len(filtered_df) * 100
        
        # 计算平均盈利和平均亏损
        wins = filtered_df[filtered_df[return_column] > return_threshold][return_column]
        losses = filtered_df[filtered_df[return_column] <= return_threshold][return_column]
        
        avg_gain = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # 计算盈亏比，添加保护机制
        if avg_loss == 0:
            # 如果亏损为0，设置盈亏比为100（表示无限大）
            profit_loss_ratio = 100
        else:
            profit_loss_ratio = abs(avg_gain / avg_loss)
        
        return f"{win_rate:.2f}%", f"{profit_loss_ratio:.2f}"
    else:
        return "N/A", "N/A"

def calculate_kelly_position(win_rate_str, pl_ratio_str):
    try:
        # 移除百分号并转换为浮点数
        win_rate = float(win_rate_str.strip('%')) / 100
        pl_ratio = float(pl_ratio_str)
        
        # 如果盈亏比非常大（接近无穷大），设置一个合理的上限
        if pl_ratio > 100:
            pl_ratio = 100
        
        # 计算凯利公式
        kelly = (win_rate * (pl_ratio + 1) - 1) / pl_ratio
        
        # 限制最大仓位为50%
        kelly = min(max(kelly, 0), 0.5)
        
        return f"{kelly*100:.2f}%" if kelly > 0 else "0.00%"
    except:
        return "N/A"

# 创建回报率>5%的胜率和盈亏比表格数据
win_rate_5pct_data = {
    'RAI Threshold': ['Below -2', 'Below -1.5', 'Below -1', 'Below 0', 'Below 0.5', 'Below 1', 'Below 1.5'],
    '3M Win Rate': [calculate_win_rate_and_ratio(-2, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(-1.5, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(-1, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(0, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(0.5, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 5)[0]],
    '3M P/L Ratio': [calculate_win_rate_and_ratio(-2, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(-1.5, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(-1, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(0, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(0.5, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 5)[1]],
    '3M Kelly Position': [],  # 将在下面填充
    '6M Win Rate': [calculate_win_rate_and_ratio(-2, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(-1.5, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(-1, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(0, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(0.5, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 5)[0]],
    '6M P/L Ratio': [calculate_win_rate_and_ratio(-2, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(-1.5, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(-1, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(0, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(0.5, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 5)[1]],
    '6M Kelly Position': [],  # 将在下面填充
    '12M Win Rate': [calculate_win_rate_and_ratio(-2, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(-1.5, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(-1, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(0, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(0.5, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(1, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 5)[0]],
    '12M P/L Ratio': [calculate_win_rate_and_ratio(-2, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(-1.5, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(-1, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(0, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(0.5, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(1, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 5)[1]],
    '12M Kelly Position': []  # 将在下面填充
}

# 计算凯利仓位
for i in range(len(win_rate_5pct_data['RAI Threshold'])):
    win_rate_5pct_data['3M Kelly Position'].append(
        calculate_kelly_position(win_rate_5pct_data['3M Win Rate'][i], win_rate_5pct_data['3M P/L Ratio'][i])
    )
    win_rate_5pct_data['6M Kelly Position'].append(
        calculate_kelly_position(win_rate_5pct_data['6M Win Rate'][i], win_rate_5pct_data['6M P/L Ratio'][i])
    )
    win_rate_5pct_data['12M Kelly Position'].append(
        calculate_kelly_position(win_rate_5pct_data['12M Win Rate'][i], win_rate_5pct_data['12M P/L Ratio'][i])
    )

# 创建回报率>10%的胜率和盈亏比表格数据
win_rate_10pct_data = {
    'RAI Threshold': ['Below -2', 'Below -1.5', 'Below -1', 'Below 0', 'Below 0.5', 'Below 1', 'Below 1.5'],
    '3M Win Rate': [calculate_win_rate_and_ratio(-2, '3M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(-1.5, '3M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(-1, '3M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(0, '3M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(0.5, '3M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(1, '3M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 10)[0]],
    '3M P/L Ratio': [calculate_win_rate_and_ratio(-2, '3M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(-1.5, '3M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(-1, '3M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(0, '3M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(0.5, '3M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(1, '3M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 10)[1]],
    '3M Kelly Position': [],  # 将在下面填充
    '6M Win Rate': [calculate_win_rate_and_ratio(-2, '6M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(-1.5, '6M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(-1, '6M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(0, '6M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(0.5, '6M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(1, '6M_Future_Return', 10)[0],
                    calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 10)[0]],
    '6M P/L Ratio': [calculate_win_rate_and_ratio(-2, '6M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(-1.5, '6M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(-1, '6M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(0, '6M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(0.5, '6M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(1, '6M_Future_Return', 10)[1],
                     calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 10)[1]],
    '6M Kelly Position': [],  # 将在下面填充
    '12M Win Rate': [calculate_win_rate_and_ratio(-2, '12M_Future_Return', 10)[0],
                     calculate_win_rate_and_ratio(-1.5, '12M_Future_Return', 10)[0],
                     calculate_win_rate_and_ratio(-1, '12M_Future_Return', 10)[0],
                     calculate_win_rate_and_ratio(0, '12M_Future_Return', 10)[0],
                     calculate_win_rate_and_ratio(0.5, '12M_Future_Return', 10)[0],
                     calculate_win_rate_and_ratio(1, '12M_Future_Return', 10)[0],
                     calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 10)[0]],
    '12M P/L Ratio': [calculate_win_rate_and_ratio(-2, '12M_Future_Return', 10)[1],
                      calculate_win_rate_and_ratio(-1.5, '12M_Future_Return', 10)[1],
                      calculate_win_rate_and_ratio(-1, '12M_Future_Return', 10)[1],
                      calculate_win_rate_and_ratio(0, '12M_Future_Return', 10)[1],
                      calculate_win_rate_and_ratio(0.5, '12M_Future_Return', 10)[1],
                      calculate_win_rate_and_ratio(1, '12M_Future_Return', 10)[1],
                      calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 10)[1]],
    '12M Kelly Position': []  # 将在下面填充
}

# 计算凯利仓位
for i in range(len(win_rate_10pct_data['RAI Threshold'])):
    win_rate_10pct_data['3M Kelly Position'].append(
        calculate_kelly_position(win_rate_10pct_data['3M Win Rate'][i], win_rate_10pct_data['3M P/L Ratio'][i])
    )
    win_rate_10pct_data['6M Kelly Position'].append(
        calculate_kelly_position(win_rate_10pct_data['6M Win Rate'][i], win_rate_10pct_data['6M P/L Ratio'][i])
    )
    win_rate_10pct_data['12M Kelly Position'].append(
        calculate_kelly_position(win_rate_10pct_data['12M Win Rate'][i], win_rate_10pct_data['12M P/L Ratio'][i])
    )

# 创建 DataFrames
win_rate_5pct_df = pd.DataFrame(win_rate_5pct_data)
win_rate_10pct_df = pd.DataFrame(win_rate_10pct_data)

def calculate_momentum_change(momentum_series, days=21):  # 21个交易日约等于1个月
    # 计算过去一个月的变化
    changes = momentum_series.diff(days)
    
    # 将变化标准化到[0,1]区间
    min_change = changes.min()
    max_change = changes.max()
    normalized_changes = (changes - min_change) / (max_change - min_change)
    
    return normalized_changes

# 计算momentum的变化
rai_df['Momentum_Change'] = calculate_momentum_change(rai_df['Momentum'])

def calculate_win_rate_and_ratio_with_momentum(rai_threshold, momentum_min, momentum_max, return_column, return_threshold, is_rai_less_than_threshold=True):
    # 合并两个 DataFrame，确保日期对齐
    merged_df = pd.merge(spx_df, rai_df, on='Date', how='inner')
    
    # 根据RAI阈值筛选数据
    if is_rai_less_than_threshold:
        filtered_df = merged_df[merged_df['Headline'] < rai_threshold]
    else:
        filtered_df = merged_df[merged_df['Headline'] >= rai_threshold]
    
    # 进一步根据momentum变化筛选数据
    filtered_df = filtered_df[
        (filtered_df['Momentum_Change'] >= momentum_min) & 
        (filtered_df['Momentum_Change'] < momentum_max)
    ]
    
    # 计算胜率和盈亏比
    if len(filtered_df) > 0 and return_column in filtered_df.columns:
        # 计算回报率大于阈值的记录数
        win_count = len(filtered_df[filtered_df[return_column] > return_threshold])
        # 计算胜率
        win_rate = win_count / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        
        # 计算平均盈利和平均亏损
        wins = filtered_df[filtered_df[return_column] > return_threshold][return_column]
        losses = filtered_df[filtered_df[return_column] <= return_threshold][return_column]
        
        avg_gain = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # 计算盈亏比，添加保护机制
        if avg_loss == 0:
            profit_loss_ratio = 100
        else:
            profit_loss_ratio = abs(avg_gain / avg_loss)
        
        return f"{win_rate:.2f}%", f"{profit_loss_ratio:.2f}"
    else:
        return "N/A", "N/A"

# 创建基于RAI和Momentum变化的胜率和盈亏比表格数据
momentum_data = {
    'Condition': [
        'RAI < -2, Momentum Change [0, 0.33)',
        'RAI < -2, Momentum Change [0.33, 0.67)',
        'RAI < -2, Momentum Change [0.67, 1)',
        'RAI < -1, Momentum Change [0, 0.33)',
        'RAI < -1, Momentum Change [0.33, 0.67)',
        'RAI < -1, Momentum Change [0.67, 1)',
        'RAI < -0.5, Momentum Change [0, 0.33)',
        'RAI < -0.5, Momentum Change [0.33, 0.67)',
        'RAI < -0.5, Momentum Change [0.67, 1)',
        'RAI < 0, Momentum Change [0, 0.33)',
        'RAI < 0, Momentum Change [0.33, 0.67)',
        'RAI < 0, Momentum Change [0.67, 1)',
        'RAI < 0.5, Momentum Change [0, 0.33)',
        'RAI < 0.5, Momentum Change [0.33, 0.67)',
        'RAI < 0.5, Momentum Change [0.67, 1)',
        'RAI < 1, Momentum Change [0, 0.33)',
        'RAI < 1, Momentum Change [0.33, 0.67)',
        'RAI < 1, Momentum Change [0.67, 1)'
    ],
    '3M Win Rate': [],
    '3M P/L Ratio': [],
    '3M Kelly Position': [],
    '6M Win Rate': [],
    '6M P/L Ratio': [],
    '6M Kelly Position': [],
    '12M Win Rate': [],
    '12M P/L Ratio': [],
    '12M Kelly Position': []
}

# 计算各种条件下的胜率和盈亏比
conditions = [
    (-2, 0, 0.33, True),
    (-2, 0.33, 0.67, True),
    (-2, 0.67, 1, True),
    (-1, 0, 0.33, True),
    (-1, 0.33, 0.67, True),
    (-1, 0.67, 1, True),
    (-0.5, 0, 0.33, True),
    (-0.5, 0.33, 0.67, True),
    (-0.5, 0.67, 1, True),
    (0, 0, 0.33, True),
    (0, 0.33, 0.67, True),
    (0, 0.67, 1, True),
    (0.5, 0, 0.33, True),
    (0.5, 0.33, 0.67, True),
    (0.5, 0.67, 1, True),
    (1, 0, 0.33, True),
    (1, 0.33, 0.67, True),
    (1, 0.67, 1, True)
]

for i, (rai_threshold, momentum_min, momentum_max, is_less_than) in enumerate(conditions):
    # 3个月
    win_rate, pl_ratio = calculate_win_rate_and_ratio_with_momentum(
        rai_threshold, momentum_min, momentum_max, '3M_Future_Return', 5, is_less_than
    )
    momentum_data['3M Win Rate'].append(win_rate)
    momentum_data['3M P/L Ratio'].append(pl_ratio)
    momentum_data['3M Kelly Position'].append(calculate_kelly_position(win_rate, pl_ratio))
    
    # 6个月
    win_rate, pl_ratio = calculate_win_rate_and_ratio_with_momentum(
        rai_threshold, momentum_min, momentum_max, '6M_Future_Return', 5, is_less_than
    )
    momentum_data['6M Win Rate'].append(win_rate)
    momentum_data['6M P/L Ratio'].append(pl_ratio)
    momentum_data['6M Kelly Position'].append(calculate_kelly_position(win_rate, pl_ratio))
    
    # 12个月
    win_rate, pl_ratio = calculate_win_rate_and_ratio_with_momentum(
        rai_threshold, momentum_min, momentum_max, '12M_Future_Return', 5, is_less_than
    )
    momentum_data['12M Win Rate'].append(win_rate)
    momentum_data['12M P/L Ratio'].append(pl_ratio)
    momentum_data['12M Kelly Position'].append(calculate_kelly_position(win_rate, pl_ratio))

# 创建 DataFrames
momentum_df = pd.DataFrame(momentum_data)

# 创建 Dash 应用
app = Dash(__name__)

# 创建 RAI 的 Plotly 图表
rai_fig = px.line(rai_df, x='Date', y='Headline', title='RAI 数据随时间变化', markers=True)

# 自定义 RAI 线图样式
rai_fig.update_traces(
    line=dict(color='#E31837', width=2),  # 设置线条颜色为 #E31837
    marker=dict(size=6, color='#E31837'),  # 设置标记颜色为 #E31837
    name='Headline'  # 明确设置图例名称为 Headline
)

# 为 RAI 图表添加跌幅超过10%的区域的浅桔色阴影
for date in drop_dates:
    rai_fig.add_vrect(
        x0=date,
        x1=date + pd.Timedelta(days=63),  # 改为63个交易日，与计算Future_3M_Return时保持一致
        fillcolor='peachpuff',  # 浅桔色
        opacity=0.3,
        layer='below',  # 确保阴影在线图下方
        line_width=0
    )

# 修改虚线水平线
for y_value in [1, -1, -2]:
    rai_fig.add_shape(
        type='line',
        x0=rai_df['Date'].min(),  # 水平线起始点 X 坐标
        x1=rai_df['Date'].max(),  # 水平线结束点 X 坐标
        y0=y_value,               # 水平线 Y 坐标
        y1=y_value,
        line=dict(color='grey', width=3, dash='dash')  # 将宽度从1增加到3
    )

# 单独为值为 0 的水平线设置更宽的线宽
rai_fig.add_shape(
    type='line',
    x0=rai_df['Date'].min(),  # 水平线起始点 X 坐标
    x1=rai_df['Date'].max(),  # 水平线结束点 X 坐标
    y0=0,                     # 水平线 Y 坐标
    y1=0,
    line=dict(color='grey', width=5, dash='dash')  # 将宽度增加到5
)

# 添加每年的垂直线（实线）
for year in range(rai_df['Date'].min().year, rai_df['Date'].max().year + 1):
    rai_fig.add_shape(
        type='line',
        x0=pd.Timestamp(f'{year}-01-01'),  # 垂直线 X 坐标（每年 1 月 1 日）
        x1=pd.Timestamp(f'{year}-01-01'),
        y0=rai_df['Headline'].min(),          # 垂直线起始点 Y 坐标
        y1=rai_df['Headline'].max(),          # 垂直线结束点 Y 坐标
        line=dict(color='grey', width=1)   # 设置实线样式（移除 dash='dash'）
    )

# 在 RAI 图表中添加最新值和日期的注释
latest_rai_value = rai_df['Headline'].iloc[-1]
latest_rai_date = rai_df['Date'].iloc[-1].strftime('%Y-%m-%d')

rai_fig.add_annotation(
    x=latest_rai_date,
    y=latest_rai_value,
    text=f"Latest Value: {latest_rai_value:.2f}<br>Date: {latest_rai_date}",  # 使用 <br> 换行
    showarrow=True,
    arrowhead=2,
    ax=0,  # 水平偏移量
    ay=40,  # 垂直偏移量（正数表示向下）
    bgcolor='white',
    bordercolor='black',
    borderwidth=1,
    font=dict(size=16, color='black'),
    xanchor='center',  # 文本水平居中
    yanchor='top',     # 文本垂直居上
    xref='x',
    yref='y',
    xshift=0,
    yshift=0
)

# 添加 Momentum 数据到图表
rai_fig.add_trace(
    go.Scatter(
        x=rai_df['Date'],  # 使用日期作为 X 轴
        y=rai_df['Momentum'],  # 使用 Momentum 数据作为 Y 轴
        mode='lines',  # 折线图
        name='Momentum',  # 图例名称
        line=dict(color='purple', width=2),  # 设置线条颜色和宽度
        yaxis='y2'  # 使用第二个 Y 轴（如果需要）
    )
)

# 如果需要第二个 Y 轴，更新布局
rai_fig.update_layout(
    yaxis2=dict(
        title='Momentum',  # 第二个 Y 轴的标题
        overlaying='y',  # 覆盖在第一个 Y 轴上
        side='right'  # 显示在右侧
    )
)

# 修改 RAI 图表的图例设置
rai_fig.update_layout(
    legend=dict(
        title='Legend',  # 图例标题
        orientation='h',  # 水平排列
        yanchor='bottom',  # 图例锚点在底部
        y=-0.3,  # 将图例放在图表下方
        xanchor='center',  # 图例水平居中
        x=0.5,
        itemsizing='constant'  # 保持图例项大小一致
    )
)

# 确保 Headline 数据在图例中显示
rai_fig.update_traces(
    name='Headline',  # 设置图例名称为 Headline
    selector=dict(name='Headline')  # 选择 Headline 数据
)

# 创建标普500的 Plotly 图表
spx_fig = px.line(spx_df, x='Date', y='Last Price', title='标普500收盘价', markers=True)

# 自定义标普500线图样式
spx_fig.update_traces(
    line=dict(color='blue', width=2),  # 设置线条颜色为蓝色
    marker=dict(size=6, color='blue')  # 设置标记颜色为蓝色
)

# 为标普500图表添加跌幅超过10%的区域的浅桔色阴影
for date in drop_dates:
    spx_fig.add_vrect(
        x0=date,
        x1=date + pd.Timedelta(days=63),  # 改为63个交易日，与计算Future_3M_Return时保持一致
        fillcolor='peachpuff',  # 浅桔色
        opacity=0.3,
        layer='below',  # 确保阴影在线图下方
        line_width=0
    )

# 在标普500图表中添加最新值和日期的注释
latest_spx_value = spx_df['Last Price'].iloc[-1]
latest_spx_date = spx_df['Date'].iloc[-1].strftime('%Y-%m-%d')

spx_fig.add_annotation(
    x=latest_spx_date,
    y=latest_spx_value,
    text=f"Latest Value: {latest_spx_value:.2f}<br>Date: {latest_spx_date}",  # 使用 <br> 换行
    showarrow=True,
    arrowhead=2,
    ax=0,  # 水平偏移量
    ay=40,  # 垂直偏移量（正数表示向下）
    bgcolor='white',
    bordercolor='black',
    borderwidth=1,
    font=dict(size=16, color='black'),
    xanchor='center',  # 文本水平居中
    yanchor='top',     # 文本垂直居上
    xref='x',
    yref='y',
    xshift=0,
    yshift=0
)

# 获取最后一天的日期
last_date = max(rai_df['Date'].max(), spx_df['Date'].max())
first_date = min(rai_df['Date'].min(), spx_df['Date'].min())

# 修改 RAI 图表布局
rai_fig.update_layout(
    xaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        range=[min_date, '2025-12-31'],  # 将 x 轴范围延长至 2025 年末
        domain=[0, 1],
        dtick='M6',
        tickformat='%Y-%m',
        tickangle=45,
        fixedrange=True,  # 固定x轴范围，防止缩放
        showspikes=True,  # 显示十字虚线的垂直线
        spikemode='across',  # 使十字虚线跨越整个图表
        spikedash='dot',  # 设置虚线样式为点状
        spikecolor='grey',  # 设置虚线颜色为灰色
        spikethickness=1  # 设置虚线宽度
    ),
    yaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        range=[-3, rai_df['Headline'].max()],
        showspikes=True,  # 显示十字虚线的水平线
        spikemode='across',  # 使十字虚线跨越整个图表
        spikedash='dot',  # 设置虚线样式为点状
        spikecolor='grey',  # 设置虚线颜色为灰色
        spikethickness=1  # 设置虚线宽度
    ),
    hovermode='x unified'  # 鼠标悬停时显示十字虚线
)

# 修改标普500图表布局
spx_fig.update_layout(
    xaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        range=[min_date, '2025-12-31'],  # 将 x 轴范围延长至 2025 年末
        domain=[0, 1],
        dtick='M6',
        tickformat='%Y-%m',
        tickangle=45,
        fixedrange=True,  # 固定x轴范围，防止缩放
        showspikes=True,  # 显示十字虚线的垂直线
        spikemode='across',  # 使十字虚线跨越整个图表
        spikedash='dot',  # 设置虚线样式为点状
        spikecolor='grey',  # 设置虚线颜色为灰色
        spikethickness=1  # 设置虚线宽度
    ),
    yaxis=dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        tickfont=dict(size=24),
        showspikes=True,  # 显示十字虚线的水平线
        spikemode='across',  # 使十字虚线跨越整个图表
        spikedash='dot',  # 设置虚线样式为点状
        spikecolor='grey',  # 设置虚线颜色为灰色
        spikethickness=1  # 设置虚线宽度
    ),
    hovermode='x unified'  # 鼠标悬停时显示十字虚线
)

# 在 RAI 图表中添加最后一天的垂直线
rai_fig.add_vline(
    x=last_date,
    line_width=2,
    line_dash="dash",
    line_color="black"
)

# 在标普500图表中添加最后一天的垂直线
spx_fig.add_vline(
    x=last_date,
    line_width=2,
    line_dash="dash",
    line_color="black"
)

# 定义布局
app.layout = html.Div(children=[
    html.H1(children='RAI 和标普500数据可视化'),
    dcc.Graph(
        id='rai-graph',
        figure=rai_fig,
        style={'height': '800px'}  # 将RAI图表高度改为与标普500图表相同
    ),
    dcc.Graph(
        id='spx-graph',
        figure=spx_fig,
        style={'height': '800px'}  # 保持标普500图表高度不变
    ),
    html.H3(children='标普500回报率>5%的胜率和盈亏比'),
    dash_table.DataTable(
        id='win-rate-5pct-table',
        columns=[{"name": i, "id": i} for i in win_rate_5pct_df.columns],
        data=win_rate_5pct_df.to_dict('records'),
        style_table={'margin': '20px'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontSize': '24px'
        },
        style_header={
            'backgroundColor': '#E8E8E8',
            'fontWeight': 'bold',
            'fontSize': '24px'
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{3M Win Rate} > 60 && {3M Win Rate} != "N/A"',
                    'column_id': '3M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{6M Win Rate} > 60 && {6M Win Rate} != "N/A"',
                    'column_id': '6M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{12M Win Rate} > 60 && {12M Win Rate} != "N/A"',
                    'column_id': '12M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            # 3个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "40.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "30.00%" && {3M Kelly Position} < "40.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "20.00%" && {3M Kelly Position} < "30.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            },
            # 6个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "40.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "30.00%" && {6M Kelly Position} < "40.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "20.00%" && {6M Kelly Position} < "30.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            },
            # 12个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "40.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "30.00%" && {12M Kelly Position} < "40.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "20.00%" && {12M Kelly Position} < "30.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            }
        ]
    ),
    html.Div([
        html.H4('凯利公式说明：'),
        html.P([
            '1. 凯利公式计算最优仓位：f* = p(b+1)-1/b，其中p为胜率，b为盈亏比',
            html.Br(),
            '2. 绿色背景表示该策略的最优仓位大小：',
            html.Br(),
            '   - 浅绿色：20%-30%',
            html.Br(),
            '   - 中绿色：30%-40%',
            html.Br(),
            '   - 深绿色：40%-50%',
            html.Br(),
            '3. 建议实际使用时取凯利公式计算结果的一半或更保守的仓位',
            html.Br(),
            '4. P/L Ratio（盈亏比）= 平均盈利/平均亏损',
            html.Br(),
            '5. 如果胜率或盈亏比数据无效，最优仓位将显示为"N/A"'
        ])
    ], style={'margin': '20px', 'fontSize': '18px'}),
    html.H3(children='标普500回报率>10%的胜率和盈亏比'),
    dash_table.DataTable(
        id='win-rate-10pct-table',
        columns=[{"name": i, "id": i} for i in win_rate_10pct_df.columns],
        data=win_rate_10pct_df.to_dict('records'),
        style_table={'margin': '20px'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontSize': '24px'
        },
        style_header={
            'backgroundColor': '#E8E8E8',
            'fontWeight': 'bold',
            'fontSize': '24px'
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{3M Win Rate} > 60 && {3M Win Rate} != "N/A"',
                    'column_id': '3M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{6M Win Rate} > 60 && {6M Win Rate} != "N/A"',
                    'column_id': '6M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{12M Win Rate} > 60 && {12M Win Rate} != "N/A"',
                    'column_id': '12M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            # 3个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "40.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "30.00%" && {3M Kelly Position} < "40.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "20.00%" && {3M Kelly Position} < "30.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            },
            # 6个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "40.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "30.00%" && {6M Kelly Position} < "40.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "20.00%" && {6M Kelly Position} < "30.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            },
            # 12个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "40.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "30.00%" && {12M Kelly Position} < "40.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "20.00%" && {12M Kelly Position} < "30.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            }
        ]
    ),
    html.Div([
        html.H4('凯利公式说明：'),
        html.P([
            '1. 凯利公式计算最优仓位：f* = p(b+1)-1/b，其中p为胜率，b为盈亏比',
            html.Br(),
            '2. 绿色背景表示该策略的最优仓位大小：',
            html.Br(),
            '   - 浅绿色：20%-30%',
            html.Br(),
            '   - 中绿色：30%-40%',
            html.Br(),
            '   - 深绿色：40%-50%',
            html.Br(),
            '3. 建议实际使用时取凯利公式计算结果的一半或更保守的仓位',
            html.Br(),
            '4. P/L Ratio（盈亏比）= 平均盈利/平均亏损',
            html.Br(),
            '5. 如果胜率或盈亏比数据无效，最优仓位将显示为"N/A"'
        ])
    ], style={'margin': '20px', 'fontSize': '18px'}),
    html.H3(children='基于RAI和Momentum变化的标普500回报率>5%的胜率和盈亏比'),
    dash_table.DataTable(
        id='momentum-table',
        columns=[{"name": i, "id": i} for i in momentum_df.columns],
        data=momentum_df.to_dict('records'),
        style_table={'margin': '20px'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
            'fontSize': '24px'
        },
        style_header={
            'backgroundColor': '#E8E8E8',
            'fontWeight': 'bold',
            'fontSize': '24px'
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{3M Win Rate} > 60 && {3M Win Rate} != "N/A"',
                    'column_id': '3M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{6M Win Rate} > 60 && {6M Win Rate} != "N/A"',
                    'column_id': '6M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{12M Win Rate} > 60 && {12M Win Rate} != "N/A"',
                    'column_id': '12M Win Rate'
                },
                'backgroundColor': '#90EE90'
            },
            # 3个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "40.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "30.00%" && {3M Kelly Position} < "40.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{3M Kelly Position} >= "20.00%" && {3M Kelly Position} < "30.00%" && {3M Kelly Position} != "N/A"',
                    'column_id': '3M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            },
            # 6个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "40.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "30.00%" && {6M Kelly Position} < "40.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{6M Kelly Position} >= "20.00%" && {6M Kelly Position} < "30.00%" && {6M Kelly Position} != "N/A"',
                    'column_id': '6M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            },
            # 12个月Kelly仓位的三个层级
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "40.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#228B22'  # 深绿色
            },
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "30.00%" && {12M Kelly Position} < "40.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#32CD32'  # 中绿色
            },
            {
                'if': {
                    'filter_query': '{12M Kelly Position} >= "20.00%" && {12M Kelly Position} < "30.00%" && {12M Kelly Position} != "N/A"',
                    'column_id': '12M Kelly Position'
                },
                'backgroundColor': '#90EE90'  # 浅绿色
            }
        ]
    )
])

# 运行应用
if __name__ == '__main__':
    print("Dash 应用已启动，请访问以下网址查看图表：")
    print("http://127.0.0.1:8050/")
    app.run_server(debug=True)

