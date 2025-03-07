import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据
data = {
    'Index': ['STOXX50'],
    '法国': [35],
    '德国': [30],
    '荷兰': [10],
    '西班牙': [8],
    '意大利': [7],
    '比利时': [3],
    '芬兰': [2],
    '爱尔兰': [2],
    '其他': [3]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 保存到Excel
df.to_excel('geo.xlsx', index=False)

# 创建堆叠条形图
plt.figure(figsize=(12, 2))

# 计算每个国家的起始位置
left = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# 绘制每个国家的条形
for i, country in enumerate(data.keys()):
    if country != 'Index':
        width = data[country][0]
        rect = plt.barh('STOXX50', width, left=left, color=colors[i-1], 
                label=f'{country} - {width}%')
        # 在条形图中心添加数值标签
        plt.text(left + width/2, 0, f'{width}%', 
                ha='center', va='center', color='white', fontweight='bold')
        left += width

# 设置图表样式
plt.xlabel('百分比')
plt.title('STOXX50 国家权重分布')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存图表
plt.tight_layout()
plt.savefig('geo_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
