import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import time

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 调整默认字体大小
rcParams['font.size'] = 22.5  # 默认字体大小再缩小10%（原为25）

# 读取 Excel 文件
df = pd.read_excel('PE.xlsx', parse_dates=['Date'], index_col='Date')

# 检查可用列名
print("可用列名：", df.columns.tolist())

# 修改时间过滤方式
end_date = pd.Timestamp.now()
start_date = end_date - pd.DateOffset(years=10)
df = df.loc[start_date:end_date]

# 需要计算的指数（根据实际列名调整）
indices = ['MSCI AC World', 'SPX', 'STOXX 50', 'STOXX 600', 'HS INDEX', 'MSCI JAPAN']

# 计算结果
results = {
    '指数': [],
    '过去10年均值': [],
    '过去10年最小值': [],
    '过去10年最大值': [],
    '当前值': []
}

for index in indices:
    if index not in df.columns:
        print(f"警告：列 '{index}' 不存在，跳过计算")
        continue
        
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
fig, ax = plt.subplots(figsize=(24, 13.5))  # 16:9比例 (24/13.5=1.777)

# 设置背景色为白色
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 设置y轴范围，确保从5开始
y_min = min(results_df['过去10年最小值']) - 1
y_min = min(y_min, 5)  # 确保y轴最小值不超过5
y_max = max(results_df['过去10年最大值']) + 2
ax.set_ylim(y_min, y_max)

# 绘制范围条形图
x = np.arange(len(results_df))  # 使用 results_df 的长度而不是 indices
bar_width = 0.6

for i in range(len(results_df)):  # 使用 results_df 的长度进行循环
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
ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=38.7)  # 缩小图例字体（原为43）

# 设置x轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels(results_df['指数'], fontsize=38.7)  # 缩小x轴标签字体（原为43）

# 设置y轴刻度
y_ticks = np.arange(y_min, y_max, 2)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{int(tick) if tick.is_integer() else tick}" for tick in y_ticks], fontsize=38.7)  # 缩小y轴标签字体（原为43）

# 添加网格线（仅y轴）
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='gray')

# 设置标题和标签
# ax.set_title('主要指数的前向市盈率比较', fontsize=32, fontweight='bold')  # 增大标题字体
ax.set_ylabel('前向市盈率', fontsize=45)  # 缩小y轴标题字体（原为50）

# 添加注释说明
plt.figtext(0.1, 0.01, 
            "数据来源: 彭博社, 截至" + df.index[-1].strftime('%Y年%m月%d日'), 
            fontsize=32.4, ha='left')  # 缩小注释字体（原为36）

# 调整布局
plt.subplots_adjust(bottom=0.15)  # 增加底部边距
plt.tight_layout(pad=3.0)  # 增加整体边距

# 保存图表
plt.savefig('PE_analysis.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
