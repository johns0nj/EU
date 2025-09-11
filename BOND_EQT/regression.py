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
    
    # 获取2025年1月至3月的数据（已删除）
    # jan_2025_start = pd.to_datetime('2025-01-01')
    # mar_2025_end = pd.to_datetime('2025-03-31')
    # jan_mar_2025_data = df[(df.iloc[:, 0] >= jan_2025_start) & (df.iloc[:, 0] <= mar_2025_end)]
    
    # 获取2025年3月20日至4月21日的数据（已删除）
    # mar20_2025_start = pd.to_datetime('2025-03-20')
    # apr21_2025_end = pd.to_datetime('2025-04-21')
    # mar_apr_2025_data = df[(df.iloc[:, 0] >= mar20_2025_start) & (df.iloc[:, 0] <= apr21_2025_end)]
    
    # 使用固定大小
    size = np.ones(len(df)) * 6.5  # 使用固定大小6.5（原来是5，增加30%）
    
    return df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, size

# 处理第一个数据集
df1, X1, Y1, reg1, Y1_pred, std_dev1, latest_X1, latest_Y1, latest_date1, size1 = process_data('reg.xlsx')

# 处理第二个数据集
df2, X2, Y2, reg2, Y2_pred, std_dev2, latest_X2, latest_Y2, latest_date2, size2 = process_data('reg_HS Tech.xlsx')

# 定义创建图表的函数
def create_figure(df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, size, title):
    fig = go.Figure()
    
    # 获取起点数据（已删除近3个月路径）
    # start_date = recent_data.iloc[0, 0]  # 获取起点日期
    # start_X = recent_data.iloc[0, 2]     # 获取起点X值
    # start_Y = recent_data.iloc[0, 1]     # 获取起点Y值
    
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
    
    # 添加近3个月的数据路径（已删除）
    # fig.add_trace(
    #     go.Scatter(
    #         x=recent_data.iloc[:, 2],  # C列作为X
    #         y=recent_data.iloc[:, 1],  # B列作为Y
    #         mode='lines+markers',
    #         name='近3个月路径',
    #         line=dict(color='deepskyblue', width=2),  # 改为深天空蓝
    #         marker=dict(size=5.6),  # 从8缩小30%到5.6
    #         hovertemplate='日期: %{text|%Y-%m-%d}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
    #         text=recent_data.iloc[:, 0]  # 添加日期信息
    #     )
    # )
    
    # 添加起点标注（已删除，因为删除了近3个月路径）
    # fig.add_trace(
    #     go.Scatter(
    #         x=[start_X],
    #         y=[start_Y],
    #         mode='markers+text',
    #         name='3个月起点',
    #         marker=dict(
    #             color='khaki',
    #             size=14,
    #             symbol='diamond'
    #         ),
    #         text=[f'起点<br>日期: {start_date.strftime("%Y-%m-%d")}<br>X: {start_X:.2f}<br>Y: {start_Y:.2f}'],
    #         textfont=dict(
    #             size=15.4,  # 从14增大10%到15.4
    #             family="Arial, sans-serif",
    #             color="black",
    #             weight="bold"  # 加粗
    #         ),
    #         textposition="bottom right",
    #         showlegend=False,
    #         hovertemplate='日期: %{text}<extra></extra>'
    #     )
    # )
    
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
    
    # 添加2025年1-3月的数据路径（已删除）
    # if len(jan_mar_2025_data) > 0:
    #     # 所有1-3月路径相关代码已删除
    
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
    
    # 计算年底前三次降息情景的新值
    rate_cut_X = latest_X + 0.4  # X值增加0.4
    rate_cut_Y = latest_Y  # Y值保持与最新值相同，不变
    
    # 标注最新值（仍有降息预期未计入）
    fig.add_trace(
        go.Scatter(
            x=[rate_cut_X],
            y=[rate_cut_Y],
            mode='markers+text',
            name='最新值（仍有降息预期未计入）',
            marker=dict(
                color='yellow',  # 亮黄色
                size=13.26,  # 与最新值相同大小
                symbol='diamond'
            ),
            text=[f'最新值（仍有降息预期未计入）<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {rate_cut_X:.2f}<br>Y: {rate_cut_Y:.2f}'],
            textfont=dict(
                size=15.4,
                family="Arial, sans-serif",
                color="black",
                weight="bold"
            ),
            textposition="top right",  # 放在图表右上方空白处
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
    
    # 计算降息预期的新值
    rate_cut_X = latest_X + 0.4  # X值增加0.4
    rate_cut_Y = latest_Y  # Y值保持与最新值相同，不变
    
    # 标注最新值（仍有降息预期未计入）
    fig.add_trace(
        go.Scatter(
            x=[rate_cut_X],
            y=[rate_cut_Y],
            mode='markers+text',
            name='最新值（仍有降息预期未计入）',
            marker=dict(
                color='yellow',  # 亮黄色
                size=13.26,  # 与最新值相同大小
                symbol='diamond'
            ),
            text=[f'最新值（仍有降息预期未计入）<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {rate_cut_X:.2f}<br>Y: {rate_cut_Y:.2f}'],
            textfont=dict(
                size=15.4,
                family="Arial, sans-serif",
                color="black",
                weight="bold"
            ),
            textposition="top right",  # 放在图表右上方空白处
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
            title="Y值：恒生科技1年前瞻估值（x）",
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

# 创建三个图表
fig1 = create_figure(df1, X1, Y1, reg1, Y1_pred, std_dev1, latest_X1, latest_Y1, latest_date1, size1, 
                     "恒生科技1年前瞻估值vs中美利差回归分析")
fig2 = create_figure(df2, X2, Y2, reg2, Y2_pred, std_dev2, latest_X2, latest_Y2, latest_date2, size2, 
                     "恒生科技1年前瞻估值vs中美利差回归分析")
# 创建简化图表（第一个数据集）
fig3 = create_simplified_figure(df1, X1, Y1, reg1, Y1_pred, std_dev1, latest_X1, latest_Y1, latest_date1,
                               "简化视图：恒生科技1年前瞻估值vs中美利差回归分析")

# 创建Dash布局
app.layout = html.Div([
    # 第一个图表
    html.Div([
        html.H3(f"第一个数据集回归方程: Y = {reg1.coef_[0]:.4f} * X + {reg1.intercept_:.4f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"最新值: X = {latest_X1:.2f}, Y = {latest_Y1:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"最新值（仍有降息预期未计入）: X = {latest_X1 + 0.4:.2f}, Y = {latest_Y1:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'orange'})
    ]),
    dcc.Graph(
        id='regression-chart-1',
        figure=fig1,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    ),
    
    # 分隔线
    html.Hr(style={'margin': '30px 0'}),
    
    # 第二个图表
    html.Div([
        html.H3(f"第二个数据集回归方程: Y = {reg2.coef_[0]:.4f} * X + {reg2.intercept_:.4f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"最新值: X = {latest_X2:.2f}, Y = {latest_Y2:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"最新值（仍有降息预期未计入）: X = {latest_X2 + 0.4:.2f}, Y = {latest_Y2:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'orange'})
    ]),
    dcc.Graph(
        id='regression-chart-2',
        figure=fig2,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    ),
    
    # 分隔线
    html.Hr(style={'margin': '30px 0'}),
    
    # 第三个图表（简化视图）
    html.Div([
        html.H3(f"简化视图 - 第一个数据集回归方程: Y = {reg1.coef_[0]:.4f} * X + {reg1.intercept_:.4f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px'}),
        html.H3(f"最新值（仍有降息预期未计入）: X = {latest_X1 + 0.4:.2f}, Y = {latest_Y1:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'orange'})
    ]),
    dcc.Graph(
        id='regression-chart-3',
        figure=fig3,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    )
])

if __name__ == '__main__':
    print("应用已启动，请访问 http://127.0.0.1:8050/")
    app.run_server(debug=True)
