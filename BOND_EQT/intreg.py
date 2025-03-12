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
def weighted_regression(data, y_column='前瞻PS'):
    X = data['未来3年收入CAGR'].values.reshape(-1, 1)
    y = data[y_column].values  # 使用传入的列名
    weights = data['市值占比'].values
    
    reg = LinearRegression()
    reg.fit(X, y, sample_weight=weights)
    y_pred = reg.predict(X)
    
    return X, y, y_pred, reg

# 进行回归分析
us_X, us_y, us_y_pred, us_reg = weighted_regression(us_data, y_column='前瞻PS')
cn_X, cn_y, cn_y_pred, cn_reg = weighted_regression(cn_data, y_column='前瞻PS')

# 创建Dash应用
app = Dash(__name__)

# 创建图表
fig = go.Figure()

# 将市值标准化到合适的点大小范围（例如5到20）
min_size = 5
max_size = 20
us_size = (us_data['市值'] - us_data['市值'].min()) / (us_data['市值'].max() - us_data['市值'].min()) * (max_size - min_size) + min_size
cn_size = (cn_data['市值'] - cn_data['市值'].min()) / (cn_data['市值'].max() - cn_data['市值'].min()) * (max_size - min_size) + min_size

# 添加美国数据
fig.add_trace(
    go.Scatter(
        x=us_X.flatten(),
        y=us_y,
        mode='markers+text',
        name='美股',
        marker=dict(
            color='blue', 
            size=us_size,  # 使用根据市值计算的大小
            sizemode='diameter'  # 确保大小按直径计算
        ),
        text=us_data['简称'],
        textposition='top center',
        textfont=dict(color='darkblue'),
        hovertemplate='简称: %{text}<br>市值占比: %{customdata:.2%}<br>未来3年收入CAGR: %{x:.2f}%<br>前瞻PS: %{y:.2f}<extra></extra>',
        customdata=us_data['市值占比'],
        cliponaxis=True  # 确保文本显示在坐标轴区域内
    )
)

# 添加美国回归线
fig.add_trace(
    go.Scatter(
        x=us_X.flatten(),
        y=us_y_pred,
        mode='lines',
        name='美股回归线',
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
        name='中国股票',
        marker=dict(
            color='red', 
            size=cn_size,  # 使用根据市值计算的大小
            sizemode='diameter'  # 确保大小按直径计算
        ),
        text=cn_data['简称'],
        textposition='top center',
        textfont=dict(color='darkred'),
        hovertemplate='简称: %{text}<br>市值占比: %{customdata:.2%}<br>未来3年收入CAGR: %{x:.2f}%<br>前瞻PS: %{y:.2f}<extra></extra>',
        customdata=cn_data['市值占比'],
        cliponaxis=True  # 确保文本显示在坐标轴区域内
    )
)

# 添加中国回归线
fig.add_trace(
    go.Scatter(
        x=cn_X.flatten(),
        y=cn_y_pred,
        mode='lines',
        name='中国股票回归线',
        line=dict(color='red', width=2),
        hovertemplate='未来3年收入CAGR: %{x:.2f}%<br>预测前瞻PS估值: %{y:.2f}<extra></extra>'
    )
)

# 更新布局
fig.update_layout(
    title="中美互联网公司未来3年收入CAGR与3年前瞻PS估值回归分析",
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

# 读取PE数据
df_pe = pd.read_excel('intreg_PE.xlsx')

# 打印整个DataFrame
print("读取的PE数据：")
print(df_pe)

# 计算市值占比
df_pe['市值占比'] = df_pe.groupby('地区')['市值'].transform(lambda x: x / x.sum())

# 分割数据
us_data_pe = df_pe[df_pe['地区'] == '美国']
cn_data_pe = df_pe[df_pe['地区'] == '中国']

# 进行回归分析
us_X_pe, us_y_pe, us_y_pred_pe, us_reg_pe = weighted_regression(us_data_pe, y_column='前瞻PE')
cn_X_pe, cn_y_pe, cn_y_pred_pe, cn_reg_pe = weighted_regression(cn_data_pe, y_column='前瞻PE')

# 创建PE图表
fig_pe = go.Figure()

# 将市值标准化到合适的点大小范围（例如5到20）
us_size_pe = (us_data_pe['市值'] - us_data_pe['市值'].min()) / (us_data_pe['市值'].max() - us_data_pe['市值'].min()) * (max_size - min_size) + min_size
cn_size_pe = (cn_data_pe['市值'] - cn_data_pe['市值'].min()) / (cn_data_pe['市值'].max() - cn_data_pe['市值'].min()) * (max_size - min_size) + min_size

# 添加美国数据
fig_pe.add_trace(
    go.Scatter(
        x=us_X_pe.flatten(),
        y=us_y_pe,
        mode='markers+text',
        name='美股',
        marker=dict(
            color='blue', 
            size=us_size_pe,  # 使用根据市值计算的大小
            sizemode='diameter'  # 确保大小按直径计算
        ),
        text=us_data_pe['简称'],
        textposition='top center',
        textfont=dict(color='darkblue'),
        hovertemplate='简称: %{text}<br>市值占比: %{customdata:.2%}<br>未来3年收入CAGR: %{x:.2f}%<br>前瞻PE: %{y:.2f}<extra></extra>',
        customdata=us_data_pe['市值占比'],
        cliponaxis=True  # 确保文本显示在坐标轴区域内
    )
)

# 添加美国回归线
fig_pe.add_trace(
    go.Scatter(
        x=us_X_pe.flatten(),
        y=us_y_pred_pe,
        mode='lines',
        name='美股回归线',
        line=dict(color='blue', width=2),
        hovertemplate='未来3年收入CAGR: %{x:.2f}%<br>预测前瞻PE估值: %{y:.2f}<extra></extra>'
    )
)

# 添加中国数据
fig_pe.add_trace(
    go.Scatter(
        x=cn_X_pe.flatten(),
        y=cn_y_pe,
        mode='markers+text',
        name='中国股票',
        marker=dict(
            color='red', 
            size=cn_size_pe,  # 使用根据市值计算的大小
            sizemode='diameter'  # 确保大小按直径计算
        ),
        text=cn_data_pe['简称'],
        textposition='top center',
        textfont=dict(color='darkred'),
        hovertemplate='简称: %{text}<br>市值占比: %{customdata:.2%}<br>未来3年收入CAGR: %{x:.2f}%<br>前瞻PE: %{y:.2f}<extra></extra>',
        customdata=cn_data_pe['市值占比'],
        cliponaxis=True  # 确保文本显示在坐标轴区域内
    )
)

# 添加中国回归线
fig_pe.add_trace(
    go.Scatter(
        x=cn_X_pe.flatten(),
        y=cn_y_pred_pe,
        mode='lines',
        name='中国股票回归线',
        line=dict(color='red', width=2),
        hovertemplate='未来3年收入CAGR: %{x:.2f}%<br>预测前瞻PE估值: %{y:.2f}<extra></extra>'
    )
)

# 更新PE图表布局
fig_pe.update_layout(
    title="中美互联网公司未来3年收入CAGR与3年前瞻PE估值回归分析",
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
        title="前瞻PE",
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
    ),
    html.H3("美国PE回归方程: Y = {:.4f} * X + {:.4f}".format(us_reg_pe.coef_[0], us_reg_pe.intercept_)),
    html.H3("中国PE回归方程: Y = {:.4f} * X + {:.4f}".format(cn_reg_pe.coef_[0], cn_reg_pe.intercept_)),
    dcc.Graph(
        id='regression-chart-pe',
        figure=fig_pe,
        config={
            'displayModeBar': True,
            'scrollZoom': True
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
