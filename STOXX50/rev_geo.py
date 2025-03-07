import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据（使用区间的中位数）
data = {
    'Index': ['STOXX50'],
    '欧洲': [45],  # 40-50%的中位数
    '北美': [25],  # 20-30%的中位数
    '亚太地区': [20],  # 15-25%的中位数
    '其他地区': [10]  # 5-10%的中位数
}

# 创建DataFrame
df = pd.DataFrame(data)

# 保存到Excel
df.to_excel('rev_geo.xlsx', index=False)

# 创建堆叠条形图
plt.figure(figsize=(12, 2))

# 计算每个地区的起始位置
left = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 为每个地区设置不同的颜色

# 绘制每个地区的条形
for i, region in enumerate(data.keys()):
    if region != 'Index':
        width = data[region][0]
        rect = plt.barh('STOXX50', width, left=left, color=colors[i-1], 
                label=f'{region} - {width}%')
        # 在条形图中心添加数值标签
        plt.text(left + width/2, 0, f'{width}%', 
                ha='center', va='center', color='white', fontweight='bold')
        left += width

# 设置图表样式
plt.xlabel('收入占比（%）')
plt.title('STOXX50 指数成分股全球收入分布')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存图表
plt.tight_layout()
plt.savefig('rev_geo_distribution.png', bbox_inches='tight', dpi=300)
plt.close() 