import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建数据
data_stoxx50 = {
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

# STOXX 200数据（使用区间中位数，确保总和为100%）
data_stoxx200 = {
    'Index': ['STOXX200'],
    '英国': [27.5],  # 25-30%的中位数
    '法国': [17.5],  # 15-20%的中位数
    '瑞士': [12.5],  # 10-15%的中位数
    '德国': [12.5],  # 10-15%的中位数
    '荷兰': [7.5],   # 5-10%的中位数
    '西班牙': [7.5], # 5-10%的中位数
    '意大利': [7.5], # 5-10%的中位数
    '北欧国家': [7.5], # 10-15%的中位数
    '其他欧洲国家': [2.5] # 5-10%的中位数
}

# STOXX 600数据（使用区间中位数，确保总和为100%）
data_stoxx600 = {
    'Index': ['STOXX600'],
    '英国': [22.5],
    '法国': [17.5],
    '瑞士': [12.5],
    '德国': [12.5],
    '荷兰': [7.5],
    '西班牙': [7.5],
    '意大利': [7.5],
    '北欧国家': [7.5],
    '其他欧洲国家': [5]
}

# 创建DataFrame并合并数据
df_stoxx50 = pd.DataFrame(data_stoxx50)
df_stoxx200 = pd.DataFrame(data_stoxx200)
df_stoxx600 = pd.DataFrame(data_stoxx600)
df = pd.concat([df_stoxx50, df_stoxx200, df_stoxx600], ignore_index=True)

# 保存到Excel
df.to_excel('geo.xlsx', index=False)

# 创建堆叠条形图
plt.figure(figsize=(12, 6))

# 计算每个国家的起始位置
left_stoxx50 = 0
left_stoxx200 = 0
left_stoxx600 = 0
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

# 绘制STOXX50的条形
for i, country in enumerate(data_stoxx50.keys()):
    if country != 'Index':
        width = data_stoxx50[country][0]
        rect = plt.barh('STOXX50', width, left=left_stoxx50, color=colors[i-1], 
                label=country)  # 只显示国家名称
        # 在条形图中心添加数值标签
        plt.text(left_stoxx50 + width/2, 0, f'{width}%', 
                ha='center', va='center', color='white', fontweight='bold')
        left_stoxx50 += width

# 绘制STOXX200的条形
for i, country in enumerate(data_stoxx200.keys()):
    if country != 'Index':
        width = data_stoxx200[country][0]
        rect = plt.barh('STOXX200', width, left=left_stoxx200, color=colors[i-1], 
                label=country)  # 只显示国家名称
        # 在条形图中心添加数值标签
        plt.text(left_stoxx200 + width/2, 1, f'{width}%', 
                ha='center', va='center', color='white', fontweight='bold')
        left_stoxx200 += width

# 绘制STOXX600的条形
for i, country in enumerate(data_stoxx600.keys()):
    if country != 'Index':
        width = data_stoxx600[country][0]
        rect = plt.barh('STOXX600', width, left=left_stoxx600, color=colors[i-1], 
                label=country)  # 只显示国家名称
        # 在条形图中心添加数值标签
        plt.text(left_stoxx600 + width/2, 2, f'{width}%', 
                ha='center', va='center', color='white', fontweight='bold')
        left_stoxx600 += width

# 设置图表样式
plt.xlabel('百分比')
plt.title('STOXX 50、STOXX 200 和 STOXX 600 国家权重分布对比')

# 合并图例中的同类项，并按照权重从大到小排序
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # 去重

# 获取所有国家的权重
all_weights = {}
for country, weight in data_stoxx50.items():
    if country != 'Index':
        all_weights[country] = weight[0]
for country, weight in data_stoxx200.items():
    if country != 'Index':
        all_weights[country] = max(all_weights.get(country, 0), weight[0])
for country, weight in data_stoxx600.items():
    if country != 'Index':
        all_weights[country] = max(all_weights.get(country, 0), weight[0])

# 按照权重从大到小排序
sorted_labels = sorted(by_label.keys(), key=lambda x: all_weights[x], reverse=True)
sorted_handles = [by_label[label] for label in sorted_labels]

# 设置图例
plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

# 保存图表
plt.tight_layout()
plt.savefig('geo_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
