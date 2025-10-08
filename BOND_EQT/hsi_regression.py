import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from sklearn.linear_model import LinearRegression

# 定义数据处理函数
def process_data(excel_file, title_suffix=""):
    # 读取数据
    df = pd.read_excel(excel_file)
    
    print(f"\n{excel_file} DataFrame的列名：")
    print(df.columns)
    print(f"\n{excel_file} DataFrame的前几行：")
    print(df.head())
    
    # 确保日期列是datetime类型 - 处理Excel日期序列号
    from datetime import datetime, timedelta
    excel_epoch = datetime(1899, 12, 30)  # Excel的起始日期
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: excel_epoch + timedelta(days=x) if isinstance(x, (int, float)) else pd.to_datetime(x))
    
    # 筛选2020年7月以来的数据
    df = df[df.iloc[:, 0] >= pd.to_datetime('2020-07-01')]  # 假设日期在第一列
    
    # 删除包含NaN值的行
    df = df.dropna()
    
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
    
    # 使用固定大小
    size = np.ones(len(df)) * 6.5  # 使用固定大小6.5（原来是5，增加30%）
    
    return df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, size

# 处理恒生指数数据集
df_hsi, X_hsi, Y_hsi, reg_hsi, Y_hsi_pred, std_dev_hsi, latest_X_hsi, latest_Y_hsi, latest_date_hsi, size_hsi = process_data('reg_HSI.xlsx')

# 计算置信水平下的回报率函数
def calculate_return_confidence(reg, std_dev, current_valuation, x_min=-3, x_max=-1, confidence_level=0.95):
    """
    计算在指定中美利差区间内，给定置信水平下的恒生指数未来1年回报率
    
    参数:
    reg: 线性回归模型
    std_dev: 残差标准差
    current_valuation: 当前估值水平
    x_min, x_max: 中美利差区间
    confidence_level: 置信水平
    
    返回:
    dict: 包含回报率统计信息
    """
    import scipy.stats as stats
    
    # 生成利差区间内的预测点
    x_range = np.linspace(x_min, x_max, 100)
    
    # 计算回归预测值
    y_pred_range = reg.predict(x_range.reshape(-1, 1))
    
    # 计算置信区间的z值
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # 计算置信区间边界
    y_upper = y_pred_range + z_score * std_dev
    y_lower = y_pred_range - z_score * std_dev
    
    # 计算回报率（相对于当前估值）
    returns_pred = (y_pred_range / current_valuation - 1) * 100
    returns_upper = (y_upper / current_valuation - 1) * 100
    returns_lower = (y_lower / current_valuation - 1) * 100
    
    # 统计信息
    result = {
        'x_range': x_range,
        'predicted_returns': returns_pred,
        'upper_bound_returns': returns_upper,
        'lower_bound_returns': returns_lower,
        'mean_return': np.mean(returns_pred),
        'upper_mean_return': np.mean(returns_upper),
        'lower_mean_return': np.mean(returns_lower),
        'return_range': (np.min(returns_lower), np.max(returns_upper))
    }
    
    return result

# 计算当前估值水平下的回报率
return_analysis = calculate_return_confidence(reg_hsi, std_dev_hsi, latest_Y_hsi)

# 计算不考虑中美利差范围限制的回报率函数
def calculate_general_return_confidence(reg, std_dev, current_valuation, confidence_level=0.95):
    """
    计算从当前估值水平出发，不考虑中美利差范围限制的95%置信区间下的未来1年回报率
    
    参数:
    reg: 线性回归模型
    std_dev: 残差标准差
    current_valuation: 当前估值水平
    confidence_level: 置信水平
    
    返回:
    dict: 包含回报率统计信息
    """
    import scipy.stats as stats
    
    # 计算置信区间的z值
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # 基于当前估值和残差标准差计算置信区间
    # 假设未来估值围绕当前估值正态分布，标准差为回归残差标准差
    upper_valuation = current_valuation + z_score * std_dev
    lower_valuation = current_valuation - z_score * std_dev
    
    # 计算回报率（相对于当前估值）
    upper_return = (upper_valuation / current_valuation - 1) * 100
    lower_return = (lower_valuation / current_valuation - 1) * 100
    
    # 统计信息
    result = {
        'current_valuation': current_valuation,
        'upper_valuation': upper_valuation,
        'lower_valuation': lower_valuation,
        'upper_return': upper_return,
        'lower_return': lower_return,
        'return_range': (lower_return, upper_return),
        'confidence_level': confidence_level * 100
    }
    
    return result

# 计算不考虑中美利差范围的回报率
general_return_analysis = calculate_general_return_confidence(reg_hsi, std_dev_hsi, latest_Y_hsi)

# 定义创建图表的函数
def create_figure(df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, size, title):
    fig = go.Figure()
    
    # 添加原始数据点
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y,
            mode='markers',
            name='原始数据',
            marker=dict(
                color='darkred',  # 改为深红色
                size=size,  # 使用固定大小
                sizemode='diameter',  # 确保大小按直径计算
                line=dict(
                    color='darkred',  # 边框也用深红色
                    width=1  # 边框宽度
                )
            ),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'  # 移除市值显示
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
    
    # 添加+2倍标准差线
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y_pred + 2 * std_dev,
            mode='lines',
            name='+2标准差',
            line=dict(color='gray', width=2, dash='dash'),  # 灰色虚线
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        )
    )
    
    # 添加-2倍标准差线
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y_pred - 2 * std_dev,
            mode='lines',
            name='-2标准差',
            line=dict(color='gray', width=2, dash='dash'),  # 灰色虚线
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
                 size=13.26,  # 从7.8增加70%到13.26
                symbol='diamond'
            ),
            text=[f'最新值<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {latest_X:.2f}<br>Y: {latest_Y:.2f}'],
            textfont=dict(
                size=15.4,
                family="Arial, sans-serif",
                color="black",
                weight="bold"
            ),
            textposition="bottom right",  # 调整位置避免与降息情景重叠
            hovertemplate='日期: %{text}<extra></extra>'
        )
    )
    
    
    # 更新布局
    fig.update_layout(
        title=title,
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
            title="Y值：恒生指数1年前瞻估值（x）",
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
        width=1200  # 16:8长宽比
    )
    
    return fig

# 定义创建简化图表的函数
def create_simplified_figure(df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, title):
    fig = go.Figure()
    
    # 添加回归线
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y_pred,
            mode='lines',
            name='回归线',
            line=dict(color='red', width=3),
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
            line=dict(color='black', width=2, dash='dot'),
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
            line=dict(color='black', width=2, dash='dot'),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        )
    )
    
    # 添加+2倍标准差线
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y_pred + 2 * std_dev,
            mode='lines',
            name='+2标准差',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        )
    )
    
    # 添加-2倍标准差线
    fig.add_trace(
        go.Scatter(
            x=X.flatten(),
            y=Y_pred - 2 * std_dev,
            mode='lines',
            name='-2标准差',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        )
    )
    
    # 计算更新后的网格加仓点位对应估值（在回归线上）
    grid_X = latest_X + 0.4  # X值增加0.4
    grid_Y = reg.predict([[grid_X]])[0]  # Y值在回归线上
    
    # 标注更新后的网格加仓点位对应估值
    fig.add_trace(
        go.Scatter(
            x=[grid_X],
            y=[grid_Y],
            mode='markers+text',
            name='更新后的网格加仓点位对应估值',
            marker=dict(
                color='blue',  # 蓝色
                size=13.26,  # 与其他点相同大小
                symbol='circle'  # 使用圆形区分
            ),
            text=[f'更新后的网格加仓点位对应估值<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {grid_X:.2f}<br>Y: {grid_Y:.2f}'],
            textfont=dict(
                size=15.4,
                family="Arial, sans-serif",
                color="black",
                weight="bold"
            ),
            textposition="bottom right",  # 避免与其他标签重叠
            hovertemplate='日期: %{text}<extra></extra>'
        )
    )
    
    # 计算考虑EPS后的网格加仓点位对应估值（回归线的95%）
    eps_X = latest_X + 0.4  # X值增加0.4
    eps_Y = reg.predict([[eps_X]])[0] * 0.95  # Y值为回归线的95%
    
    # 标注考虑EPS后的网格加仓点位对应估值
    fig.add_trace(
        go.Scatter(
            x=[eps_X],
            y=[eps_Y],
            mode='markers+text',
            name='考虑EPS后的网格加仓点位对应估值',
            marker=dict(
                color='green',  # 绿色
                size=13.26,  # 与其他点相同大小
                symbol='square'  # 使用方形区分
            ),
            text=[f'考虑EPS后的网格加仓点位对应估值<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {eps_X:.2f}<br>Y: {eps_Y:.2f}'],
            textfont=dict(
                size=15.4,
                family="Arial, sans-serif",
                color="black",
                weight="bold"
            ),
            textposition="bottom left",  # 避免与其他标签重叠
            hovertemplate='日期: %{text}<extra></extra>'
        )
    )
    
    # 更新布局
    fig.update_layout(
        title=title,
        title_font=dict(size=24),
        xaxis=dict(
            title="X值：中美10年期国债收益率利差（%）",
            title_font=dict(size=22),
            tickfont=dict(size=17),
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title="Y值：恒生指数1年前瞻估值（x）",
            title_font=dict(size=22),
            tickfont=dict(size=17),
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        legend=dict(
            font=dict(size=17)
        ),
        showlegend=True,
        plot_bgcolor='white',
        height=600,
        width=1200
    )
    
    return fig

# 创建Dash应用
app = Dash(__name__)

# 创建图表
fig_hsi = create_figure(df_hsi, X_hsi, Y_hsi, reg_hsi, Y_hsi_pred, std_dev_hsi, latest_X_hsi, latest_Y_hsi, latest_date_hsi, size_hsi, 
                       "恒生指数1年前瞻估值vs中美利差回归分析")

# 创建简化图表
fig_hsi_simplified = create_simplified_figure(df_hsi, X_hsi, Y_hsi, reg_hsi, Y_hsi_pred, std_dev_hsi, latest_X_hsi, latest_Y_hsi, latest_date_hsi,
                                             "简化视图：恒生指数1年前瞻估值vs中美利差回归分析")

# 创建Dash布局
app.layout = html.Div([
    # 恒生指数完整图表
    html.Div([
        html.H3(f"恒生指数回归方程: Y = {reg_hsi.coef_[0]:.4f} * X + {reg_hsi.intercept_:.4f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"最新值: X = {latest_X_hsi:.2f}, Y = {latest_Y_hsi:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"95%置信水平下，中美利差在-3到-1区间内，恒生指数未来1年回报率区间: {return_analysis['return_range'][0]:.1f}% ~ {return_analysis['return_range'][1]:.1f}%", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '20px', 'color': 'blue'}),
        html.H3(f"平均预期回报率: {return_analysis['mean_return']:.1f}% (上界: {return_analysis['upper_mean_return']:.1f}%, 下界: {return_analysis['lower_mean_return']:.1f}%)", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '18px', 'color': 'green'}),
        html.H3(f"95%置信水平下，不考虑中美利差范围限制，恒生指数未来1年回报率区间: {general_return_analysis['return_range'][0]:.1f}% ~ {general_return_analysis['return_range'][1]:.1f}%", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '20px', 'color': 'purple'}),
        html.H3(f"当前估值: {general_return_analysis['current_valuation']:.2f}x, 上界估值: {general_return_analysis['upper_valuation']:.2f}x, 下界估值: {general_return_analysis['lower_valuation']:.2f}x", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '18px', 'color': 'darkred'})
    ]),
    dcc.Graph(
        id='hsi-regression-chart',
        figure=fig_hsi,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    ),
    
    # 分隔线
    html.Hr(style={'margin': '30px 0'}),
    
    # 恒生指数简化图表
    html.Div([
        html.H3(f"简化视图 - 恒生指数回归方程: Y = {reg_hsi.coef_[0]:.4f} * X + {reg_hsi.intercept_:.4f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"更新后的网格加仓点位对应估值: X = {latest_X_hsi + 0.4:.2f}, Y = {reg_hsi.predict([[latest_X_hsi + 0.4]])[0]:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'blue'}),
        html.H3(f"考虑EPS后的网格加仓点位对应估值: X = {latest_X_hsi + 0.4:.2f}, Y = {reg_hsi.predict([[latest_X_hsi + 0.4]])[0] * 0.95:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'green'})
    ]),
    dcc.Graph(
        id='hsi-regression-chart-simplified',
        figure=fig_hsi_simplified,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    )
])

if __name__ == '__main__':
    print("恒生指数回归分析应用已启动，请访问 http://127.0.0.1:8050/")
    app.run_server(debug=True)
