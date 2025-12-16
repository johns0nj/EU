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
    
    # 查找2025年10月2日的数据
    target_date = pd.to_datetime('2025-10-02')
    oct_2_data = df[df.iloc[:, 0] == target_date]
    if len(oct_2_data) > 0:
        oct_2_X = oct_2_data.iloc[0, 2]  # C列作为X
        oct_2_Y = oct_2_data.iloc[0, 1]  # B列作为Y
    else:
        oct_2_X = None
        oct_2_Y = None
    
    # 查找2025年4月8日的数据
    apr_8_date = pd.to_datetime('2025-04-08')
    apr_8_data = df[df.iloc[:, 0] == apr_8_date]
    if len(apr_8_data) > 0:
        apr_8_X = apr_8_data.iloc[0, 2]  # C列作为X
        apr_8_Y = apr_8_data.iloc[0, 1]  # B列作为Y
    else:
        apr_8_X = None
        apr_8_Y = None
    
    # 查找2025年4月9日的数据
    apr_9_date = pd.to_datetime('2025-04-09')
    apr_9_data = df[df.iloc[:, 0] == apr_9_date]
    if len(apr_9_data) > 0:
        apr_9_X = apr_9_data.iloc[0, 2]  # C列作为X
        apr_9_Y = apr_9_data.iloc[0, 1]  # B列作为Y
    else:
        apr_9_X = None
        apr_9_Y = None
    
    # 查找X=0.68, Y=16.74的特定数据点
    # 使用容差来匹配浮点数
    tolerance = 0.01
    specific_data = df[
        (abs(df.iloc[:, 2] - 0.68) < tolerance) & 
        (abs(df.iloc[:, 1] - 16.74) < tolerance)
    ]
    if len(specific_data) > 0:
        specific_date = specific_data.iloc[0, 0]
        specific_X = specific_data.iloc[0, 2]
        specific_Y = specific_data.iloc[0, 1]
        print(f"找到匹配数据: 日期={specific_date.strftime('%Y-%m-%d')}, X={specific_X:.2f}, Y={specific_Y:.2f}")
    else:
        specific_date = None
        specific_X = None
        specific_Y = None
        print("未找到X=0.68, Y=16.74的匹配数据")
    
    # 筛选2024年1月到2月的数据
    jan_2024_start = pd.to_datetime('2024-01-01')
    feb_2024_end = pd.to_datetime('2024-02-29')
    jan_feb_2024_data = df[(df.iloc[:, 0] >= jan_2024_start) & (df.iloc[:, 0] <= feb_2024_end)]
    
    if len(jan_feb_2024_data) > 0:
        jan_feb_X = jan_feb_2024_data.iloc[:, 2].values  # C列作为X
        jan_feb_Y = jan_feb_2024_data.iloc[:, 1].values  # B列作为Y
        jan_feb_dates = jan_feb_2024_data.iloc[:, 0].values  # 日期
        print(f"找到2024年1-2月数据: {len(jan_feb_2024_data)}个数据点")
    else:
        jan_feb_X = None
        jan_feb_Y = None
        jan_feb_dates = None
        print("未找到2024年1-2月数据")
    
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
    
    return df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, size, oct_2_X, oct_2_Y, apr_8_X, apr_8_Y, apr_9_X, apr_9_Y, specific_date, specific_X, specific_Y, jan_feb_X, jan_feb_Y, jan_feb_dates

# 处理第一个数据集
df1, X1, Y1, reg1, Y1_pred, std_dev1, latest_X1, latest_Y1, latest_date1, size1, oct_2_X1, oct_2_Y1, apr_8_X1, apr_8_Y1, apr_9_X1, apr_9_Y1, specific_date1, specific_X1, specific_Y1, jan_feb_X1, jan_feb_Y1, jan_feb_dates1 = process_data('reg.xlsx')

# 处理第二个数据集
df2, X2, Y2, reg2, Y2_pred, std_dev2, latest_X2, latest_Y2, latest_date2, size2, oct_2_X2, oct_2_Y2, apr_8_X2, apr_8_Y2, apr_9_X2, apr_9_Y2, specific_date2, specific_X2, specific_Y2, jan_feb_X2, jan_feb_Y2, jan_feb_dates2 = process_data('reg_HS Tech.xlsx')

# 定义创建图表的函数
def create_figure(df, X, Y, reg, Y_pred, std_dev, latest_X, latest_Y, latest_date, size, title, oct_2_X=None, oct_2_Y=None, apr_8_X=None, apr_8_Y=None, apr_9_X=None, apr_9_Y=None, specific_date=None, specific_X=None, specific_Y=None, jan_feb_X=None, jan_feb_Y=None, jan_feb_dates=None):
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
                color='red',  # 改为正红色
                size=size,  # 使用固定大小
                sizemode='diameter',  # 确保大小按直径计算
                line=dict(
                    color='red',  # 边框也用正红色
                    width=1  # 边框宽度
                )
            ),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'  # 移除市值显示
        )
    )
    
    # 添加2024年1-2月数据的特殊标记（如果存在）
    if jan_feb_X is not None and jan_feb_Y is not None:
        fig.add_trace(
            go.Scatter(
                x=jan_feb_X,
                y=jan_feb_Y,
                mode='markers',
                name='2024年1-2月',
                marker=dict(
                    color='lightblue',  # 浅蓝色
                    size=8,  # 稍大一些
                    symbol='circle',
                    line=dict(
                        color='darkblue',  # 深蓝色边框
                        width=2
                    ),
                    opacity=0.8  # 半透明效果
                ),
                hovertemplate='2024年1-2月<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            )
        )
        
        # 添加阴影区域来突出2024年1-2月的范围
        if len(jan_feb_X) > 0:
            # 计算X和Y的范围
            x_min, x_max = min(jan_feb_X), max(jan_feb_X)
            y_min, y_max = min(jan_feb_Y), max(jan_feb_Y)
            
            # 添加矩形阴影区域
            fig.add_shape(
                type="rect",
                x0=x_min - 0.05, y0=y_min - 0.5,
                x1=x_max + 0.05, y1=y_max + 0.5,
                fillcolor="lightblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            )
            
            # 添加区域标签
            fig.add_annotation(
                x=(x_min + x_max) / 2,
                y=y_max + 0.3,
                text="2024年1-2月区域",
                showarrow=False,
                font=dict(
                    size=14,
                    color="darkblue",
                    weight="bold"
                ),
                bgcolor="lightblue",
                bordercolor="darkblue",
                borderwidth=1,
                opacity=0.8
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
                color='darkgreen',  # 深绿色
                size=13.26,  # 从7.8增加70%到13.26
                symbol='triangle-up'  # 三角形向上
            ),
            text=[f'最新值<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {latest_X:.2f}<br>Y: {latest_Y:.2f}'],
            textfont=dict(
                size=15.4,
                family="Arial, sans-serif",
                color="black",
                weight="bold"
            ),
            textposition="bottom right",  # 调整位置避免与其他标记重叠
            hovertemplate='日期: %{text}<extra></extra>'
        )
    )
    
    # 标注2025年10月2日的值（如果存在）
    if oct_2_X is not None and oct_2_Y is not None:
        fig.add_trace(
            go.Scatter(
                x=[oct_2_X],
                y=[oct_2_Y],
                mode='markers+text',
                name='2025-10-02',
                marker=dict(
                    color='blue',
                    size=15,
                    symbol='circle',
                    line=dict(color='darkblue', width=2)
                ),
                text=[f'2025-10-02<br>X: {oct_2_X:.2f}<br>Y: {oct_2_Y:.2f}'],
                textfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                    color="blue",
                    weight="bold"
                ),
                textposition="top center",
                hovertemplate='日期: 2025-10-02<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            )
        )
    
    # 标注2025年4月8日的值（如果存在）
    if apr_8_X is not None and apr_8_Y is not None:
        fig.add_trace(
            go.Scatter(
                x=[apr_8_X],
                y=[apr_8_Y],
                mode='markers+text',
                name='2025-04-08',
                marker=dict(
                    color='purple',
                    size=15,
                    symbol='square',
                    line=dict(color='darkmagenta', width=2)
                ),
                text=[f'2025-04-08<br>X: {apr_8_X:.2f}<br>Y: {apr_8_Y:.2f}'],
                textfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                    color="purple",
                    weight="bold"
                ),
                textposition="top left",
                hovertemplate='日期: 2025-04-08<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            )
        )
    
    # 标注2025年4月9日的值（如果存在）
    if apr_9_X is not None and apr_9_Y is not None:
        fig.add_trace(
            go.Scatter(
                x=[apr_9_X],
                y=[apr_9_Y],
                mode='markers+text',
                name='2025-04-09',
                marker=dict(
                    color='orange',
                    size=15,
                    symbol='diamond',
                    line=dict(color='darkorange', width=2)
                ),
                text=[f'2025-04-09<br>X: {apr_9_X:.2f}<br>Y: {apr_9_Y:.2f}'],
                textfont=dict(
                    size=14,
                    family="Arial, sans-serif",
                    color="orange",
                    weight="bold"
                ),
                textposition="top right",
                hovertemplate='日期: 2025-04-09<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            )
        )
    
    # 标注特定数据点X=0.68, Y=16.74（如果存在）
    if specific_X is not None and specific_Y is not None and specific_date is not None:
        fig.add_trace(
            go.Scatter(
                x=[specific_X],
                y=[specific_Y],
                mode='markers+text',
                name=f'{specific_date.strftime("%Y-%m-%d")}',
                marker=dict(
                    color='red',
                    size=18,
                    symbol='star',
                    line=dict(color='darkred', width=3)
                ),
                text=[f'{specific_date.strftime("%Y-%m-%d")}<br>X: {specific_X:.2f}<br>Y: {specific_Y:.2f}'],
                textfont=dict(
                    size=16,
                    family="Arial, sans-serif",
                    color="red",
                    weight="bold"
                ),
                textposition="middle right",
                hovertemplate=f'日期: {specific_date.strftime("%Y-%m-%d")}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
            )
        )
    
    # 计算年底前三次降息情景的新值
    rate_cut_X = latest_X + 0.4  # X值增加0.4
    rate_cut_Y = latest_Y  # Y值保持与最新值相同，不变
    
    # 标注最新值 - 已删除
    
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
    
    # 标注最新值
    fig.add_trace(
        go.Scatter(
            x=[rate_cut_X],
            y=[rate_cut_Y],
            mode='markers+text',
            name='最新值',
            marker=dict(
                color='yellow',  # 亮黄色
                size=13.26,  # 与最新值相同大小
                symbol='diamond'
            ),
            text=[f'最新值<br>日期: {latest_date.strftime("%Y-%m-%d")}<br>X: {rate_cut_X:.2f}<br>Y: {rate_cut_Y:.2f}'],
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
    
    # 计算更新后的网格加仓点位对应估值（在回归线上）
    grid_X = rate_cut_X  # X值与降息预期点相同
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
    eps_X = rate_cut_X  # X值与降息预期点相同
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
                     "恒生科技1年前瞻估值vs中美利差回归分析", oct_2_X1, oct_2_Y1, apr_8_X1, apr_8_Y1, apr_9_X1, apr_9_Y1, specific_date1, specific_X1, specific_Y1, jan_feb_X1, jan_feb_Y1, jan_feb_dates1)
fig2 = create_figure(df2, X2, Y2, reg2, Y2_pred, std_dev2, latest_X2, latest_Y2, latest_date2, size2, 
                     "恒生科技1年前瞻估值vs中美利差回归分析", oct_2_X2, oct_2_Y2, apr_8_X2, apr_8_Y2, apr_9_X2, apr_9_Y2, specific_date2, specific_X2, specific_Y2, jan_feb_X2, jan_feb_Y2, jan_feb_dates2)
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
        html.H3(f"最新值: X = {latest_X1 + 0.4:.2f}, Y = {latest_Y1:.2f}", 
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
        html.H3(f"最新值: X = {latest_X2 + 0.4:.2f}, Y = {latest_Y2:.2f}", 
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
        html.H3(f"最新值: X = {latest_X1 + 0.4:.2f}, Y = {latest_Y1:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'orange'}),
        html.H3(f"更新后的网格加仓点位对应估值: X = {latest_X1 + 0.4:.2f}, Y = {reg1.predict([[latest_X1 + 0.4]])[0]:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'blue'}),
        html.H3(f"考虑EPS后的网格加仓点位对应估值: X = {latest_X1 + 0.4:.2f}, Y = {reg1.predict([[latest_X1 + 0.4]])[0] * 0.95:.2f}", 
                style={'margin': '10px', 'textAlign': 'center', 'fontSize': '24px', 'color': 'green'})
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
