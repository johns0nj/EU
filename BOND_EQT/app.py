# 添加必要的导入
from dash import Dash
import dash_html_components as html

# 已知数据
current_price = 5615
current_rai = 0
target_price_4800 = 4800
target_rai_4800 = -2

# 计算斜率
slope = (target_rai_4800 - current_rai) / (target_price_4800 - current_price)

# 计算不同点位对应的RAI
def calculate_rai(price):
    return slope * (price - current_price) + current_rai

# 计算结果
prices = [5200, 5100, 5000, 4900]
rai_values = {price: calculate_rai(price) for price in prices}

# 初始化Dash应用
app = Dash(__name__)

# 在Dash布局中添加显示部分
app.layout = html.Div(children=[
    # ... 其他布局代码 ...
    
    # 添加标普500点位与RAI对应关系
    html.Div([
        html.H4('标普500点位与RAI对应关系', style={'color': '#E31837', 'marginTop': '30px', 'borderBottom': '2px solid #E31837', 'paddingBottom': '10px', 'fontSize': '31px'}),
        html.P([
            '基于当前标普500点位5615（RAI=0）和4800点位（RAI=-2）的线性关系计算：',
            html.Br(),
            html.Br(),
            *[html.Div([
                f'• 标普500点位 {price} 对应的RAI值约为: {rai_values[price]:.2f}',
                html.Br()
            ]) for price in prices],
            html.Br(),
            '说明：',
            html.Br(),
            '• 该计算基于线性插值法，假设RAI与标普500点位呈线性关系',
            html.Br(),
            '• 实际RAI值可能因市场情绪和其他因素而有所不同'
        ], style={'fontSize': '23px', 'lineHeight': '1.6'})
    ], style={'margin': '40px', 'padding': '30px', 'border': '2px solid #ddd', 'borderRadius': '20px', 'backgroundColor': '#ffffff'})
    
    # ... 其他布局代码 ...
])

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)