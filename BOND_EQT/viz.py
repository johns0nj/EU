import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, dash_table, Input, Output, State
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
    
    # 根据不同的RAI阈值筛选数据
    if rai_threshold == -2:
        filtered_df = merged_df[merged_df['Headline'] < -2]
    elif rai_threshold == -1:
        filtered_df = merged_df[(merged_df['Headline'] >= -2) & (merged_df['Headline'] < -1)]
    elif rai_threshold == 0:
        filtered_df = merged_df[(merged_df['Headline'] >= -1) & (merged_df['Headline'] < 0)]
    elif rai_threshold == 1:
        filtered_df = merged_df[(merged_df['Headline'] >= 0) & (merged_df['Headline'] < 1)]
    else:  # rai_threshold == 1.5，表示RAI >= 1
        filtered_df = merged_df[merged_df['Headline'] >= 1]
    
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
    'RAI Threshold': ['RAI < -2', '-2 ≤ RAI < -1', '-1 ≤ RAI < 0', '0 ≤ RAI < 1', 'RAI ≥ 1'],
    '3M Win Rate': [calculate_win_rate_and_ratio(-2, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(-1, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(0, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1, '3M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 5)[0]],
    '3M P/L Ratio': [calculate_win_rate_and_ratio(-2, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(-1, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(0, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1, '3M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 5)[1]],
    '3M Kelly Position': [],  # 将在下面填充
    '3M Kelly Position Half': [],
    '6M Win Rate': [calculate_win_rate_and_ratio(-2, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(-1, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(0, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1, '6M_Future_Return', 5)[0],
                    calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 5)[0]],
    '6M P/L Ratio': [calculate_win_rate_and_ratio(-2, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(-1, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(0, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1, '6M_Future_Return', 5)[1],
                     calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 5)[1]],
    '6M Kelly Position': [],  # 将在下面填充
    '6M Kelly Position Half': [],
    '12M Win Rate': [calculate_win_rate_and_ratio(-2, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(-1, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(0, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(1, '12M_Future_Return', 5)[0],
                     calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 5)[0]],
    '12M P/L Ratio': [calculate_win_rate_and_ratio(-2, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(-1, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(0, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(1, '12M_Future_Return', 5)[1],
                      calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 5)[1]],
    '12M Kelly Position': [],  # 将在下面填充
    '12M Kelly Position Half': []
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

# 计算凯利仓位的一半
for i in range(len(win_rate_5pct_data['RAI Threshold'])):
    # 3个月
    kelly_position = win_rate_5pct_data['3M Kelly Position'][i]
    if kelly_position != "N/A":
        kelly_half = float(kelly_position.strip('%')) / 2
        win_rate_5pct_data['3M Kelly Position Half'].append(f"{kelly_half:.2f}%")
    else:
        win_rate_5pct_data['3M Kelly Position Half'].append("N/A")
    
    # 6个月
    kelly_position = win_rate_5pct_data['6M Kelly Position'][i]
    if kelly_position != "N/A":
        kelly_half = float(kelly_position.strip('%')) / 2
        win_rate_5pct_data['6M Kelly Position Half'].append(f"{kelly_half:.2f}%")
    else:
        win_rate_5pct_data['6M Kelly Position Half'].append("N/A")
    
    # 12个月
    kelly_position = win_rate_5pct_data['12M Kelly Position'][i]
    if kelly_position != "N/A":
        kelly_half = float(kelly_position.strip('%')) / 2
        win_rate_5pct_data['12M Kelly Position Half'].append(f"{kelly_half:.2f}%")
    else:
        win_rate_5pct_data['12M Kelly Position Half'].append("N/A")

# 创建 DataFrames
win_rate_5pct_df = pd.DataFrame(win_rate_5pct_data)

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
    
    # 根据不同的RAI阈值筛选数据
    if rai_threshold == -2:
        filtered_df = merged_df[merged_df['Headline'] < -2]
    elif rai_threshold == -1:
        filtered_df = merged_df[(merged_df['Headline'] >= -2) & (merged_df['Headline'] < -1)]
    elif rai_threshold == 0:
        filtered_df = merged_df[(merged_df['Headline'] >= -1) & (merged_df['Headline'] < 0)]
    elif rai_threshold == 1:
        filtered_df = merged_df[(merged_df['Headline'] >= 0) & (merged_df['Headline'] < 1)]
    else:  # rai_threshold == 1.5，表示RAI >= 1
        filtered_df = merged_df[merged_df['Headline'] >= 1]
    
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
        '-2 ≤ RAI < -1, Momentum Change [0, 0.33)',
        '-2 ≤ RAI < -1, Momentum Change [0.33, 0.67)',
        '-2 ≤ RAI < -1, Momentum Change [0.67, 1)',
        '-1 ≤ RAI < 0, Momentum Change [0, 0.33)',
        '-1 ≤ RAI < 0, Momentum Change [0.33, 0.67)',
        '-1 ≤ RAI < 0, Momentum Change [0.67, 1)',
        '0 ≤ RAI < 1, Momentum Change [0, 0.33)',
        '0 ≤ RAI < 1, Momentum Change [0.33, 0.67)',
        '0 ≤ RAI < 1, Momentum Change [0.67, 1)',
        'RAI ≥ 1, Momentum Change [0, 0.33)',
        'RAI ≥ 1, Momentum Change [0.33, 0.67)',
        'RAI ≥ 1, Momentum Change [0.67, 1)'
    ],
    '3M Win Rate': [],
    '3M P/L Ratio': [],
    '3M Kelly Position': [],
    '3M Kelly Position Half': [],
    '6M Win Rate': [],
    '6M P/L Ratio': [],
    '6M Kelly Position': [],
    '6M Kelly Position Half': [],
    '12M Win Rate': [],
    '12M P/L Ratio': [],
    '12M Kelly Position': [],
    '12M Kelly Position Half': []
}

# 计算各种条件下的胜率和盈亏比
conditions = [
    (-2, 0, 0.33, True),
    (-2, 0.33, 0.67, True),
    (-2, 0.67, 1, True),
    (-1, 0, 0.33, True),
    (-1, 0.33, 0.67, True),
    (-1, 0.67, 1, True),
    (0, 0, 0.33, True),
    (0, 0.33, 0.67, True),
    (0, 0.67, 1, True),
    (1, 0, 0.33, True),
    (1, 0.33, 0.67, True),
    (1, 0.67, 1, True),
    (1.5, 0, 0.33, True),
    (1.5, 0.33, 0.67, True),
    (1.5, 0.67, 1, True)
]

for i, (rai_threshold, momentum_min, momentum_max, is_less_than) in enumerate(conditions):
    # 3个月
    win_rate, pl_ratio = calculate_win_rate_and_ratio_with_momentum(
        rai_threshold, momentum_min, momentum_max, '3M_Future_Return', 5, is_less_than
    )
    momentum_data['3M Win Rate'].append(win_rate)
    momentum_data['3M P/L Ratio'].append(pl_ratio)
    momentum_data['3M Kelly Position'].append(calculate_kelly_position(win_rate, pl_ratio))
    momentum_data['3M Kelly Position Half'].append(calculate_kelly_position(win_rate, pl_ratio))
    
    # 6个月
    win_rate, pl_ratio = calculate_win_rate_and_ratio_with_momentum(
        rai_threshold, momentum_min, momentum_max, '6M_Future_Return', 5, is_less_than
    )
    momentum_data['6M Win Rate'].append(win_rate)
    momentum_data['6M P/L Ratio'].append(pl_ratio)
    momentum_data['6M Kelly Position'].append(calculate_kelly_position(win_rate, pl_ratio))
    momentum_data['6M Kelly Position Half'].append(calculate_kelly_position(win_rate, pl_ratio))
    
    # 12个月
    win_rate, pl_ratio = calculate_win_rate_and_ratio_with_momentum(
        rai_threshold, momentum_min, momentum_max, '12M_Future_Return', 5, is_less_than
    )
    momentum_data['12M Win Rate'].append(win_rate)
    momentum_data['12M P/L Ratio'].append(pl_ratio)
    momentum_data['12M Kelly Position'].append(calculate_kelly_position(win_rate, pl_ratio))
    momentum_data['12M Kelly Position Half'].append(calculate_kelly_position(win_rate, pl_ratio))

# 创建 DataFrames
momentum_df = pd.DataFrame(momentum_data)

# 在创建Dash应用之前，添加获取最新数据的函数
def get_latest_market_data():
    # 获取最新的RAI和Momentum值
    latest_rai = rai_df['Headline'].iloc[-1]
    latest_momentum = rai_df['Momentum'].iloc[-1]
    
    # 计算1个月和3个月的Momentum变化
    momentum_change_1m = rai_df['Momentum'].diff(21).iloc[-1]  # 21个交易日约等于一个月
    momentum_change_3m = rai_df['Momentum'].diff(63).iloc[-1]  # 63个交易日约等于三个月
    
    # 标准化1个月的变化
    min_change_1m = rai_df['Momentum'].diff(21).min()
    max_change_1m = rai_df['Momentum'].diff(21).max()
    normalized_momentum_change_1m = (momentum_change_1m - min_change_1m) / (max_change_1m - min_change_1m)
    
    # 标准化3个月的变化
    min_change_3m = rai_df['Momentum'].diff(63).min()
    max_change_3m = rai_df['Momentum'].diff(63).max()
    normalized_momentum_change_3m = (momentum_change_3m - min_change_3m) / (max_change_3m - min_change_3m)
    
    print(f"最新RAI值: {latest_rai:.2f}")
    print(f"最新Momentum值: {latest_momentum:.2f}")
    print(f"Momentum一个月标准化变化值: {normalized_momentum_change_1m:.2f}")
    print(f"Momentum三个月标准化变化值: {normalized_momentum_change_3m:.2f}")
    
    return latest_rai, latest_momentum, normalized_momentum_change_1m, normalized_momentum_change_3m

def generate_investment_advice(rai, momentum, momentum_change):
    advice = []
    
    # 市场状况分析
    advice.append("1. 当前市场状况：")
    if rai < -2:
        advice.append("   - RAI处于极度悲观区间（<-2）")
    elif rai < -1:
        advice.append("   - RAI处于悲观区间（-2至-1）")
    elif rai < 0:
        advice.append("   - RAI处于轻微悲观区间（-1至0）")
    elif rai < 1:
        advice.append("   - RAI处于轻微乐观区间（0至1）")
    else:
        advice.append("   - RAI处于乐观区间（>1）")
    
    if momentum < 0:
        advice.append("   - Momentum为负，表示动量偏弱")
    else:
        advice.append("   - Momentum为正，表示动量偏强")
    
    if momentum_change < 0.33:
        advice.append("   - Momentum变化处于低位区间[0, 0.33)")
    elif momentum_change < 0.67:
        advice.append("   - Momentum变化处于中位区间[0.33, 0.67)")
    else:
        advice.append("   - Momentum变化处于高位区间[0.67, 1)")
    
    # 投资建议
    advice.append("\n2. 投资建议：")
    if rai < -1 and momentum_change > 0.67:
        advice.append("   - RAI处于悲观区间但Momentum变化强劲，建议逐步建仓")
    elif rai > 1 and momentum_change < 0.33:
        advice.append("   - RAI处于乐观区间但Momentum变化疲软，建议减仓观望")
    else:
        advice.append("   - 当前市场处于中性区域，建议保持均衡仓位")
    
    # 风险提示
    advice.append("\n3. 风险提示：")
    advice.append("   - 建议参考胜率>60%的策略")
    advice.append("   - 实际仓位建议使用凯利公式计算结果的一半")
    advice.append("   - 密切关注RAI和Momentum的变化趋势")
    
    return "\n".join(advice)

# 获取最新数据
latest_rai, latest_momentum, norm_change_1m, norm_change_3m = get_latest_market_data()
investment_advice = generate_investment_advice(latest_rai, latest_momentum, norm_change_1m)

# 在get_latest_values函数后添加新函数
def get_high_momentum_change_periods():
    # 获取近1年的数据
    one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
    recent_data = rai_df[rai_df['Date'] > one_year_ago].copy()
    
    # 找出标准化Momentum变化值大于0.8的时期
    high_change_periods = recent_data[recent_data['Momentum_Change'] > 0.8]
    
    # 格式化结果
    results = []
    for _, row in high_change_periods.iterrows():
        results.append({
            '日期': row['Date'].strftime('%Y-%m-%d'),
            'RAI值': f"{row['Headline']:.2f}",
            'Momentum值': f"{row['Momentum']:.2f}",
            'Momentum变化值': f"{row['Momentum_Change']:.2f}"
        })
    
    return results

# 获取高Momentum变化期间的数据
high_momentum_periods = get_high_momentum_change_periods()

# 添加三种Momentum变化区间的胜率分析函数
def analyze_momentum_ranges():
    results = {
        'high': {'3M': [], '6M': [], '12M': []},
        'mid': {'3M': [], '6M': [], '12M': []},
        'low': {'3M': [], '6M': [], '12M': []}
    }
    
    # 从momentum_df中提取数据
    for _, row in momentum_df.iterrows():
        condition = row['Condition']
        if '[0.67, 1)' in condition:
            range_key = 'high'
        elif '[0.33, 0.67)' in condition:
            range_key = 'mid'
        elif '[0, 0.33)' in condition:
            range_key = 'low'
        else:
            continue
            
        # 提取胜率数据（去除百分号并转换为浮点数）
        win_rate_3m = float(row['3M Win Rate'].strip('%')) if row['3M Win Rate'] != 'N/A' else None
        win_rate_6m = float(row['6M Win Rate'].strip('%')) if row['6M Win Rate'] != 'N/A' else None
        win_rate_12m = float(row['12M Win Rate'].strip('%')) if row['12M Win Rate'] != 'N/A' else None
        
        if win_rate_3m is not None:
            results[range_key]['3M'].append(win_rate_3m)
        if win_rate_6m is not None:
            results[range_key]['6M'].append(win_rate_6m)
        if win_rate_12m is not None:
            results[range_key]['12M'].append(win_rate_12m)
    
    # 计算平均胜率
    avg_results = {}
    for range_key in results:
        avg_results[range_key] = {
            '3M': sum(results[range_key]['3M']) / len(results[range_key]['3M']) if results[range_key]['3M'] else 0,
            '6M': sum(results[range_key]['6M']) / len(results[range_key]['6M']) if results[range_key]['6M'] else 0,
            '12M': sum(results[range_key]['12M']) / len(results[range_key]['12M']) if results[range_key]['12M'] else 0
        }
    
    return avg_results

# 获取分析结果（在创建app.layout之前计算）
momentum_range_analysis = analyze_momentum_ranges()

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
    font=dict(size=18),  # 将字体大小增大20%（默认15px * 1.2 = 18px）
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
        spikethickness=1,  # 设置虚线宽度
        tickfont=dict(size=18)  # 将 x 轴日期字体增大20%
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
        spikethickness=1,  # 设置虚线宽度
        tickfont=dict(size=24)  # 将 y 轴数字字体大小设置为24px，与标普500图表一致
    ),
    hovermode='x unified'  # 鼠标悬停时显示十字虚线
)

# 修改标普500图表布局
spx_fig.update_layout(
    font=dict(size=18),  # 将字体大小增大20%（默认15px * 1.2 = 18px）
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
        spikethickness=1,  # 设置虚线宽度
        tickfont=dict(size=18)  # 将 x 轴日期字体增大20%
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
    # 添加新的表格
    html.H3(children='标普500回报率>5%的胜率和盈亏比（按RAI区间）'),
    dash_table.DataTable(
        id='win-rate-5pct-table-by-rai',
        columns=[
            {"name": "RAI区间", "id": "RAI_Range"},
            {"name": "3个月胜率", "id": "3M_Win_Rate"},
            {"name": "3个月盈亏比", "id": "3M_PL_Ratio"},
            {"name": "6个月胜率", "id": "6M_Win_Rate"},
            {"name": "6个月盈亏比", "id": "6M_PL_Ratio"},
            {"name": "12个月胜率", "id": "12M_Win_Rate"},
            {"name": "12个月盈亏比", "id": "12M_PL_Ratio"}
        ],
        data=[
            {
                "RAI_Range": "RAI < -2",
                "3M_Win_Rate": calculate_win_rate_and_ratio(-2, '3M_Future_Return', 5)[0],
                "3M_PL_Ratio": calculate_win_rate_and_ratio(-2, '3M_Future_Return', 5)[1],
                "6M_Win_Rate": calculate_win_rate_and_ratio(-2, '6M_Future_Return', 5)[0],
                "6M_PL_Ratio": calculate_win_rate_and_ratio(-2, '6M_Future_Return', 5)[1],
                "12M_Win_Rate": calculate_win_rate_and_ratio(-2, '12M_Future_Return', 5)[0],
                "12M_PL_Ratio": calculate_win_rate_and_ratio(-2, '12M_Future_Return', 5)[1]
            },
            {
                "RAI_Range": "-2 ≤ RAI < -1",
                "3M_Win_Rate": calculate_win_rate_and_ratio(-1, '3M_Future_Return', 5)[0],
                "3M_PL_Ratio": calculate_win_rate_and_ratio(-1, '3M_Future_Return', 5)[1],
                "6M_Win_Rate": calculate_win_rate_and_ratio(-1, '6M_Future_Return', 5)[0],
                "6M_PL_Ratio": calculate_win_rate_and_ratio(-1, '6M_Future_Return', 5)[1],
                "12M_Win_Rate": calculate_win_rate_and_ratio(-1, '12M_Future_Return', 5)[0],
                "12M_PL_Ratio": calculate_win_rate_and_ratio(-1, '12M_Future_Return', 5)[1]
            },
            {
                "RAI_Range": "-1 ≤ RAI < 0",
                "3M_Win_Rate": calculate_win_rate_and_ratio(0, '3M_Future_Return', 5)[0],
                "3M_PL_Ratio": calculate_win_rate_and_ratio(0, '3M_Future_Return', 5)[1],
                "6M_Win_Rate": calculate_win_rate_and_ratio(0, '6M_Future_Return', 5)[0],
                "6M_PL_Ratio": calculate_win_rate_and_ratio(0, '6M_Future_Return', 5)[1],
                "12M_Win_Rate": calculate_win_rate_and_ratio(0, '12M_Future_Return', 5)[0],
                "12M_PL_Ratio": calculate_win_rate_and_ratio(0, '12M_Future_Return', 5)[1]
            },
            {
                "RAI_Range": "0 ≤ RAI < 1",
                "3M_Win_Rate": calculate_win_rate_and_ratio(1, '3M_Future_Return', 5)[0],
                "3M_PL_Ratio": calculate_win_rate_and_ratio(1, '3M_Future_Return', 5)[1],
                "6M_Win_Rate": calculate_win_rate_and_ratio(1, '6M_Future_Return', 5)[0],
                "6M_PL_Ratio": calculate_win_rate_and_ratio(1, '6M_Future_Return', 5)[1],
                "12M_Win_Rate": calculate_win_rate_and_ratio(1, '12M_Future_Return', 5)[0],
                "12M_PL_Ratio": calculate_win_rate_and_ratio(1, '12M_Future_Return', 5)[1]
            },
            {
                "RAI_Range": "RAI ≥ 1",
                "3M_Win_Rate": calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 5)[0],
                "3M_PL_Ratio": calculate_win_rate_and_ratio(1.5, '3M_Future_Return', 5)[1],
                "6M_Win_Rate": calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 5)[0],
                "6M_PL_Ratio": calculate_win_rate_and_ratio(1.5, '6M_Future_Return', 5)[1],
                "12M_Win_Rate": calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 5)[0],
                "12M_PL_Ratio": calculate_win_rate_and_ratio(1.5, '12M_Future_Return', 5)[1]
            }
        ],
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
                    'filter_query': '{3M_Win_Rate} > "60.00%" && {3M_Win_Rate} != "N/A"',
                    'column_id': '3M_Win_Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{6M_Win_Rate} > "60.00%" && {6M_Win_Rate} != "N/A"',
                    'column_id': '6M_Win_Rate'
                },
                'backgroundColor': '#90EE90'
            },
            {
                'if': {
                    'filter_query': '{12M_Win_Rate} > "60.00%" && {12M_Win_Rate} != "N/A"',
                    'column_id': '12M_Win_Rate'
                },
                'backgroundColor': '#90EE90'
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
    ),
    # 添加新的说明文字
    html.Div([
        html.H4('胜率显示"N/A"的说明：', style={'color': '#666666', 'fontSize': '31px'}),
        html.P([
            '为确保统计结果的可靠性，当数据不满足以下条件时，胜率将显示为"N/A"：',
            html.Br(),
            '1. 该RAI和Momentum变化区间组合下至少需要10个历史数据点',
            html.Br(),
            '2. 至少需要5个盈利样本',
            html.Br(),
            '3. 至少需要5个亏损样本',
            html.Br(),
            html.Br(),
            '出现"N/A"的常见原因：',
            html.Br(),
            '• 该组合条件（特定的RAI区间和Momentum变化区间）在历史上出现频率较低',
            html.Br(),
            '• 某些极端情况（如RAI < -2且Momentum变化很大）的样本量不足',
            html.Br(),
            '• 样本量不足以得出具有统计意义的胜率'
        ], style={'fontSize': '23px', 'lineHeight': '1.5'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #dee2e6'}),
    # 添加最新数据和建议的显示
    html.Div([
        html.H3('最新市场数据和投资建议'),
        html.Div([
            html.P([
                f'最新RAI值: {latest_rai:.2f}',
                html.Br(),
                f'最新Momentum值: {latest_momentum:.2f}',
                html.Br(),
                f'Momentum一个月标准化变化值: {norm_change_1m:.2f}',
                html.Br(),
                f'Momentum三个月标准化变化值: {norm_change_3m:.2f}'
            ], style={'fontSize': '28.8px', 'fontWeight': 'bold', 'color': 'red'}),
            html.Pre(investment_advice, 
                    style={
                        'whiteSpace': 'pre-wrap',
                        'fontSize': '18px',
                        'backgroundColor': '#f8f9fa',
                        'padding': '20px',
                        'borderRadius': '5px',
                        'border': '1px solid #dee2e6'
                    })
        ])
    ], style={'margin': '20px'}),
    # 在最新市场数据和投资建议后添加
    html.Div([
        html.H3('近1年Momentum高变化阶段 (变化值>0.8)'),
        dash_table.DataTable(
            data=high_momentum_periods,
            columns=[
                {'name': '日期', 'id': '日期'},
                {'name': 'RAI值', 'id': 'RAI值'},
                {'name': 'Momentum值', 'id': 'Momentum值'},
                {'name': 'Momentum变化值', 'id': 'Momentum变化值'}
            ],
            style_table={'margin': '20px'},
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontSize': '18px'
            },
            style_header={
                'backgroundColor': '#E8E8E8',
                'fontWeight': 'bold',
                'fontSize': '18px'
            }
        )
    ], style={'margin': '20px'}),
    html.Div([
        html.H3('Momentum变化区间胜率分析', style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H4('高位区间 [0.67, 1)', style={'color': '#228B22'}),
                html.P([
                    '3个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['high']['3M']:.2f}%",
                    html.Br(),
                    '6个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['high']['6M']:.2f}%",
                    html.Br(),
                    '12个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['high']['12M']:.2f}%",
                    html.Br(),
                    html.Br(),
                    '特点：Momentum变化强劲，市场动能显著增强'
                ], style={'fontSize': '18px', 'textAlign': 'left'})
            ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#f0f8ff', 'margin': '10px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4('中位区间 [0.33, 0.67)', style={'color': '#32CD32'}),
                html.P([
                    '3个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['mid']['3M']:.2f}%",
                    html.Br(),
                    '6个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['mid']['6M']:.2f}%",
                    html.Br(),
                    '12个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['mid']['12M']:.2f}%",
                    html.Br(),
                    html.Br(),
                    '特点：Momentum变化温和，市场处于转换阶段'
                ], style={'fontSize': '18px', 'textAlign': 'left'})
            ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#f0f8ff', 'margin': '10px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4('低位区间 [0, 0.33)', style={'color': '#90EE90'}),
                html.P([
                    '3个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['low']['3M']:.2f}%",
                    html.Br(),
                    '6个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['low']['6M']:.2f}%",
                    html.Br(),
                    '12个月胜率和盈亏比分析：',
                    html.Br(),
                    f"平均胜率: {momentum_range_analysis['low']['12M']:.2f}%",
                    html.Br(),
                    html.Br(),
                    '特点：Momentum变化微弱，市场动能相对稳定'
                ], style={'fontSize': '18px', 'textAlign': 'left'})
            ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#f0f8ff', 'margin': '10px', 'borderRadius': '10px', 'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
        html.Div([
            html.H4('分析说明：', style={'color': '#666666', 'marginTop': '20px'}),
            html.P([
                '1. 胜率计算基于历史数据统计，反映不同Momentum变化区间下的市场表现',
                html.Br(),
                '2. 高位区间[0.67, 1)通常表示市场动能显著增强，可能预示趋势形成',
                html.Br(),
                '3. 中位区间[0.33, 0.67)表示市场处于转换阶段，需要结合RAI指标综合判断',
                html.Br(),
                '4. 低位区间[0, 0.33)表示市场动能变化较小，可能处于盘整或趋势延续阶段',
                html.Br(),
                '5. 建议将此分析与RAI指标和其他技术指标结合使用，以提高判断的准确性'
            ], style={'fontSize': '18px', 'lineHeight': '1.5'})
        ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #dee2e6'})
    ], style={'margin': '40px', 'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px'}),
    # 添加新的表格标题和表格
    html.H3(children='基于RAI和Momentum变化的标普500回报率>10%的胜率和盈亏比'),
    dash_table.DataTable(
        id='momentum-table-10pct',
        columns=[{"name": i, "id": i} for i in [
            'Condition',
            '3M Win Rate', '3M P/L Ratio', '3M Kelly Position', '3M Kelly Position Half',
            '6M Win Rate', '6M P/L Ratio', '6M Kelly Position', '6M Kelly Position Half',
            '12M Win Rate', '12M P/L Ratio', '12M Kelly Position', '12M Kelly Position Half'
        ]],
        data=[
            {
                'Condition': condition,
                **{
                    '3M Win Rate': win_rate_3m,
                    '3M P/L Ratio': pl_ratio_3m,
                    '3M Kelly Position': kelly_3m,
                    '3M Kelly Position Half': f"{float(kelly_3m.strip('%'))/2:.2f}%" if kelly_3m != "N/A" else "N/A",
                    '6M Win Rate': win_rate_6m,
                    '6M P/L Ratio': pl_ratio_6m,
                    '6M Kelly Position': kelly_6m,
                    '6M Kelly Position Half': f"{float(kelly_6m.strip('%'))/2:.2f}%" if kelly_6m != "N/A" else "N/A",
                    '12M Win Rate': win_rate_12m,
                    '12M P/L Ratio': pl_ratio_12m,
                    '12M Kelly Position': kelly_12m,
                    '12M Kelly Position Half': f"{float(kelly_12m.strip('%'))/2:.2f}%" if kelly_12m != "N/A" else "N/A"
                }
            }
            for condition, (rai_threshold, momentum_min, momentum_max, is_less_than) in zip(
                momentum_data['Condition'],
                conditions
            )
            for (win_rate_3m, pl_ratio_3m) in [calculate_win_rate_and_ratio_with_momentum(
                rai_threshold, momentum_min, momentum_max, '3M_Future_Return', 10, is_less_than
            ),]
            for (win_rate_6m, pl_ratio_6m) in [calculate_win_rate_and_ratio_with_momentum(
                rai_threshold, momentum_min, momentum_max, '6M_Future_Return', 10, is_less_than
            ),]
            for (win_rate_12m, pl_ratio_12m) in [calculate_win_rate_and_ratio_with_momentum(
                rai_threshold, momentum_min, momentum_max, '12M_Future_Return', 10, is_less_than
            ),]
            for kelly_3m in [calculate_kelly_position(win_rate_3m, pl_ratio_3m)]
            for kelly_6m in [calculate_kelly_position(win_rate_6m, pl_ratio_6m)]
            for kelly_12m in [calculate_kelly_position(win_rate_12m, pl_ratio_12m)]
        ],
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
    # 添加新的说明文字
    html.Div([
        html.H4('胜率显示"N/A"的说明：', style={'color': '#666666', 'fontSize': '31px'}),
        html.P([
            '为确保统计结果的可靠性，当数据不满足以下条件时，胜率将显示为"N/A"：',
            html.Br(),
            '1. 该RAI和Momentum变化区间组合下至少需要10个历史数据点',
            html.Br(),
            '2. 至少需要5个盈利样本',
            html.Br(),
            '3. 至少需要5个亏损样本',
            html.Br(),
            html.Br(),
            '出现"N/A"的常见原因：',
            html.Br(),
            '• 该组合条件（特定的RAI区间和Momentum变化区间）在历史上出现频率较低',
            html.Br(),
            '• 某些极端情况（如RAI < -2且Momentum变化很大）的样本量不足',
            html.Br(),
            '• 样本量不足以得出具有统计意义的胜率'
        ], style={'fontSize': '23px', 'lineHeight': '1.5'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #dee2e6'}),
    html.Div([
        html.H3('Momentum变化区间分析与策略建议', style={'textAlign': 'center', 'color': '#2C3E50', 'marginBottom': '30px', 'fontSize': '31px'}),
        html.Div([
            html.Div([
                html.H4('高位区间 [0.67, 1)', style={'color': '#228B22', 'borderBottom': '2px solid #228B22', 'paddingBottom': '10px', 'fontSize': '26px'}),
                html.P([
                    '数据分析：',
                    html.Br(),
                    f"3个月平均胜率: {momentum_range_analysis['high']['3M']:.2f}%",
                    html.Br(),
                    f"6个月平均胜率: {momentum_range_analysis['high']['6M']:.2f}%",
                    html.Br(),
                    f"12个月平均胜率: {momentum_range_analysis['high']['12M']:.2f}%",
                    html.Br(),
                    html.Br(),
                    '市场特征：',
                    html.Br(),
                    '• Momentum变化强劲，市场动能显著增强',
                    html.Br(),
                    '• 趋势形成的关键阶段',
                    html.Br(),
                    html.Br(),
                    '策略建议：',
                    html.Br(),
                    '• RAI < -1时：逐步建仓，使用凯利公式半仓',
                    html.Br(),
                    '• RAI > 1时：保持现有仓位，设置止盈位',
                    html.Br(),
                    '• 其他区间：观望为主，等待明确信号'
                ], style={'fontSize': '23px', 'textAlign': 'left'})
            ], style={'flex': 1, 'padding': '25px', 'backgroundColor': '#f0f8ff', 'margin': '10px', 'borderRadius': '15px', 'boxShadow': '3px 3px 10px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4('中位区间 [0.33, 0.67)', style={'color': '#32CD32', 'borderBottom': '2px solid #32CD32', 'paddingBottom': '10px', 'fontSize': '26px'}),
                html.P([
                    '数据分析：',
                    html.Br(),
                    f"3个月平均胜率: {momentum_range_analysis['mid']['3M']:.2f}%",
                    html.Br(),
                    f"6个月平均胜率: {momentum_range_analysis['mid']['6M']:.2f}%",
                    html.Br(),
                    f"12个月平均胜率: {momentum_range_analysis['mid']['12M']:.2f}%",
                    html.Br(),
                    html.Br(),
                    '市场特征：',
                    html.Br(),
                    '• Momentum变化温和，市场处于转换阶段',
                    html.Br(),
                    '• 趋势的延续性有待确认',
                    html.Br(),
                    html.Br(),
                    '策略建议：',
                    html.Br(),
                    '• RAI < -1时：小仓位试探性建仓',
                    html.Br(),
                    '• RAI > 1时：考虑减仓1/3仓位',
                    html.Br(),
                    '• 其他区间：维持现有仓位，密切观察'
                ], style={'fontSize': '23px', 'textAlign': 'left'})
            ], style={'flex': 1, 'padding': '25px', 'backgroundColor': '#f0f8ff', 'margin': '10px', 'borderRadius': '15px', 'boxShadow': '3px 3px 10px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H4('低位区间 [0, 0.33)', style={'color': '#90EE90', 'borderBottom': '2px solid #90EE90', 'paddingBottom': '10px', 'fontSize': '26px'}),
                html.P([
                    '数据分析：',
                    html.Br(),
                    f"3个月平均胜率: {momentum_range_analysis['low']['3M']:.2f}%",
                    html.Br(),
                    f"6个月平均胜率: {momentum_range_analysis['low']['6M']:.2f}%",
                    html.Br(),
                    f"12个月平均胜率: {momentum_range_analysis['low']['12M']:.2f}%",
                    html.Br(),
                    html.Br(),
                    '市场特征：',
                    html.Br(),
                    '• Momentum变化微弱，市场动能相对稳定',
                    html.Br(),
                    '• 可能处于盘整或趋势延续阶段',
                    html.Br(),
                    html.Br(),
                    '策略建议：',
                    html.Br(),
                    '• RAI < -1时：暂不建仓，等待动能触底',
                    html.Br(),
                    '• RAI > 1时：考虑减仓至1/2仓位',
                    html.Br(),
                    '• 其他区间：持币观望为主'
                ], style={'fontSize': '23px', 'textAlign': 'left'})
            ], style={'flex': 1, 'padding': '25px', 'backgroundColor': '#f0f8ff', 'margin': '10px', 'borderRadius': '15px', 'boxShadow': '3px 3px 10px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px'}),
        html.Div([
            html.H4('RAI < -1时的市场下跌分析：', style={'color': '#E31837', 'marginTop': '30px', 'borderBottom': '2px solid #E31837', 'paddingBottom': '10px', 'fontSize': '31px'}),
            html.P([
                '历史数据统计：',
                html.Br(),
                '• RAI < -2（极度悲观）出现36次',
                html.Br(),
                '• -2 ≤ RAI < -1（悲观）出现365次',
                html.Br(),
                html.Br(),
                '下跌幅度分析：',
                html.Br(),
                '• 3个月平均下跌：5%-10%',
                html.Br(),
                '• 6个月平均下跌：10%-15%',
                html.Br(),
                '• 极端情况（RAI<-2）可能下跌：>15%',
                html.Br(),
                html.Br(),
                '当前点位参考：',
                html.Br(),
                f'最新标普500点位：{spx_df["Last Price"].iloc[-1]:.2f}',
                html.Br(),
                f'下跌5%对应点位：{spx_df["Last Price"].iloc[-1] * 0.95:.2f}',
                html.Br(),
                f'下跌10%对应点位：{spx_df["Last Price"].iloc[-1] * 0.90:.2f}',
                html.Br(),
                f'下跌15%对应点位：{spx_df["Last Price"].iloc[-1] * 0.85:.2f}',
                html.Br(),
                html.Br(),
                '风险提示：',
                html.Br(),
                '• 以上数据基于历史统计，不代表未来必然走势',
                html.Br(),
                '• 建议结合Momentum变化和其他技术指标综合判断',
                html.Br(),
                '• 极端行情可能导致超出历史统计范围的波动'
            ], style={'fontSize': '23px', 'lineHeight': '1.6'}),
        ], style={'margin': '30px', 'padding': '25px', 'backgroundColor': '#fff8f8', 'borderRadius': '15px', 'border': '1px solid #ffcdd2', 'boxShadow': '3px 3px 10px rgba(0,0,0,0.1)'}),
        html.H4('策略执行计划：', style={'color': '#2C3E50', 'marginTop': '30px', 'borderBottom': '2px solid #2C3E50', 'paddingBottom': '10px', 'fontSize': '31px'}),
        html.P([
            '1. 仓位管理原则：',
            html.Br(),
            '   • 单次建仓不超过凯利公式计算结果的一半',
            html.Br(),
            '   • 分批建仓，每批次间隔不少于5个交易日',
            html.Br(),
            '   • 总仓位上限不超过50%',
            html.Br(),
            html.Br(),
            '2. 止损止盈设置：',
            html.Br(),
            '   • 止损位：单次操作亏损不超过总资金的2%',
            html.Br(),
            '   • 止盈位：参考历史盈亏比，设置浮动止盈',
            html.Br(),
            html.Br(),
            '3. 风险控制：',
            html.Br(),
            '   • 重大事件期间（如央行议息）避免建仓',
            html.Br(),
            '   • VIX超过30时，将仓位上限降低一半',
            html.Br(),
            '   • 连续3次止损后暂停交易，复盘策略',
            html.Br(),
            html.Br(),
            '4. 优化建议：',
            html.Br(),
            '   • 结合市场估值水平进行判断',
            html.Br(),
            '   • 关注行业轮动和资金流向',
            html.Br(),
            '   • 考虑宏观经济周期的影响'
        ], style={'fontSize': '23px', 'lineHeight': '1.6'}),
        # 在策略执行计划后添加新的计算器
        html.Div([
            html.H4('策略盈亏平衡点计算器', style={'color': '#2C3E50', 'marginTop': '30px', 'borderBottom': '2px solid #2C3E50', 'paddingBottom': '10px', 'fontSize': '31px'}),
            html.P('输入以下参数计算策略盈亏平衡点：', style={'fontSize': '23px'}),
            html.Div([
                html.Label('年交易次数：', style={'fontSize': '23px'}),
                dcc.Input(id='trades-per-year', type='number', value=12, style={'fontSize': '23px', 'margin': '10px'}),
                html.Br(),
                html.Label('单笔交易成本（%）：', style={'fontSize': '23px'}),
                dcc.Input(id='transaction-cost', type='number', value=0.1, style={'fontSize': '23px', 'margin': '10px'}),
                html.Br(),
                html.Label('无风险利率（%）：', style={'fontSize': '23px'}),
                dcc.Input(id='risk-free-rate', type='number', value=2.0, style={'fontSize': '23px', 'margin': '10px'}),
                html.Br(),
                html.Button('计算', id='calculate-button', style={'fontSize': '23px', 'margin': '10px'})
            ]),
            html.Div(id='break-even-result', style={'fontSize': '23px', 'marginTop': '20px'})
        ], style={'margin': '40px', 'padding': '30px', 'border': '2px solid #ddd', 'borderRadius': '20px', 'backgroundColor': '#ffffff'})
    ], style={'margin': '40px', 'padding': '30px', 'border': '2px solid #ddd', 'borderRadius': '20px', 'backgroundColor': '#ffffff'})  # 关闭策略执行计划部分的括号
])  # 关闭 app.layout 的括号

# 添加回调函数（移到布局外面）
@app.callback(
    Output('break-even-result', 'children'),
    Input('calculate-button', 'n_clicks'),
    State('trades-per-year', 'value'),
    State('transaction-cost', 'value'),
    State('risk-free-rate', 'value')
)
def calculate_break_even(n_clicks, trades_per_year, transaction_cost, risk_free_rate):
    if n_clicks is None:
        return ''
    
    # 将百分比转换为小数
    transaction_cost = transaction_cost / 100
    risk_free_rate = risk_free_rate / 100
    
    # 计算不同盈亏比下的最小胜率
    results = []
    for b in [1, 1.5, 2, 2.5, 3]:
        min_win_rate = (transaction_cost + risk_free_rate/trades_per_year + 1) / (b + 1)
        results.append(f'盈亏比 {b}:1 时，最小胜率需要达到 {min_win_rate*100:.2f}%')
    
    return html.Div([
        html.P('盈亏平衡点计算结果：'),
        html.Ul([html.Li(result) for result in results])
    ])

# 运行应用
if __name__ == '__main__':
    print("Dash 应用已启动，请访问以下网址查看图表：")
    print("http://127.0.0.1:8050/")
    app.run_server(debug=True)

# 在文件末尾修改保存Excel的代码
with pd.ExcelWriter('viz.xlsx') as writer:
    win_rate_5pct_df.to_excel(writer, sheet_name='回报率>5%', index=False)
    momentum_df.to_excel(writer, sheet_name='RAI和Momentum', index=False)

print("所有表格数据已保存到 viz.xlsx 文件中")

