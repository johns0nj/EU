import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 增大默认字体大小
rcParams['font.size'] = 14  # 默认字体大小增大

# 读取 Excel 文件
df = pd.read_excel('PE.xlsx', parse_dates=['Date'], index_col='Date')

# 选择过去10年的数据
df = df.last('10Y')

# 需要计算的指数
indices = ['MSCI AC World', 'STOXX 50', 'MSCI China', 'SPX', 'MSCI Japan']

# 计算结果
results = {
    '指数': [],
    '过去10年均值': [],
    '过去10年最小值': [],
    '过去10年最大值': [],
    '当前值': []
}

for index in indices:
    # 计算均值
    mean = df[index].mean()
    # 获取最小值和最大值
    min_val = df[index].min()
    max_val = df[index].max()
    # 获取当前值（最新值）
    current = df[index].iloc[-1]
    
    # 存储结果
    results['指数'].append(index)
    results['过去10年均值'].append(mean)
    results['过去10年最小值'].append(min_val)
    results['过去10年最大值'].append(max_val)
    results['当前值'].append(current)

# 转换为 DataFrame
results_df = pd.DataFrame(results)

# 打印结果
print(results_df)

# 绘制图表
fig, ax = plt.subplots(figsize=(14, 10))  # 增大图表尺寸

# 设置背景色为白色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 设置y轴范围，确保从5开始
y_min = min(results_df['过去10年最小值']) - 1
y_min = min(y_min, 5)  # 确保y轴最小值不超过5
y_max = max(results_df['过去10年最大值']) + 2
ax.set_ylim(y_min, y_max)

# 绘制范围条形图
x = np.arange(len(indices))
bar_width = 0.6

for i, idx in enumerate(indices):
    # 绘制10年范围的浅蓝色条形
    min_val = results_df.loc[i, '过去10年最小值']
    max_val = results_df.loc[i, '过去10年最大值']
    ax.bar(x[i], max_val - min_val, bottom=min_val, width=bar_width, 
           color='#A9CCE3', alpha=0.9, edgecolor='none')
    
    # 绘制均值的深蓝色水平线
    mean_val = results_df.loc[i, '过去10年均值']
    ax.plot([x[i]-bar_width/2, x[i]+bar_width/2], [mean_val, mean_val], 
            color='#002147', linewidth=3)  # 增加线宽
    
    # 绘制当前值的浅绿色点，改为钻石形状并增大一倍
    current_val = results_df.loc[i, '当前值']
    ax.scatter(x[i], current_val, color='#58D68D', s=320, marker='D', zorder=5)  # 改为钻石形状(D)并增大大小

# 添加图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#A9CCE3', lw=16, label='10年范围'),  # 增大线宽
    Line2D([0], [0], color='#002147', lw=4, label='平均值'),  # 增大线宽
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#58D68D', 
           markersize=15, label='当前值')  # 改为钻石形状(D)
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=24)  # 增大图例字体

# 设置x轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(indices, fontsize=24)  # 增大x轴标签字体

# 设置y轴刻度
y_ticks = np.arange(y_min, y_max, 2)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{int(tick) if tick.is_integer() else tick}" for tick in y_ticks], fontsize=24)  # 增大y轴标签字体

# 添加网格线（仅y轴）
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='gray')

# 设置标题和标签
# ax.set_title('主要指数的前向市盈率比较', fontsize=32, fontweight='bold')  # 增大标题字体
ax.set_ylabel('前向市盈率', fontsize=28)  # 增大y轴标题字体

# 添加注释说明
plt.figtext(0.1, 0.01, 
            "数据来源: 彭博社, 截至" + df.index[-1].strftime('%Y年%m月%d日'), 
            fontsize=20, ha='left')  # 增大注释字体

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)

# 保存图表
plt.savefig('PE_analysis.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
