import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams.update({'font.size': 24})  # 设置基础字体大小

# 读取Excel文件
df = pd.read_excel('TAA.xlsx')

# 设置固定的国家顺序（从上到下）
country_order = ['其他国家', '中国', '日本', '欧洲', '美国']  # 反转顺序，因为matplotlib的barh从下到上绘制

# 创建图表，调整子图高度比例
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 12),  # 16:8比例 (24/12=2)
                                   gridspec_kw={'height_ratios': [5, 1, 1]})  # 高度比例为5:1:1

# 设置颜色方案
colors = {
    '美国': '#A4BED1',
    '欧洲': '#00A3DF',  # 与TAA组合权重建议中的欧洲颜色一致
    '日本': '#0072CE',
    '中国': '#003B5C',
    '其他国家': '#739CBF'
}

# 1. MSCI AC World权重图（保留所有国家）
benchmark_data = df.set_index('国家').loc[country_order].reset_index()
bars1 = ax1.barh(benchmark_data['国家'], benchmark_data['MSCI AC World权重'], 
                 color=[colors[country] for country in benchmark_data['国家']])
ax1.set_title('MSCI AC World权重分布 (%)', pad=20, fontsize=32)
# 添加数值标签
for bar in bars1:
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', ha='left', va='center', fontsize=24)

# 2. TAA组合权重图（仅显示欧洲）
taa_data = df[df['国家'] == '欧洲']
bars2 = ax2.barh(taa_data['国家'], taa_data['TAA组合国家权重'], color=colors['欧洲'])
ax2.set_title('境外TAA组合权重建议 (%)', pad=20, fontsize=32)  # 修改标题
# 添加数值标签
for bar in bars2:
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', ha='left', va='center', fontsize=24)

# 3. 差异图（仅显示欧洲）
df['差异'] = df['TAA组合国家权重'] - df['MSCI AC World权重']
diff_data = df[df['国家'] == '欧洲']
bars3 = ax3.barh(diff_data['国家'], diff_data['差异'], 
                 color=['#FF4B4B' if x < 0 else '#4CAF50' for x in diff_data['差异']])
ax3.set_title('权重差异 (%)', pad=20, fontsize=32)  # 修改标题，删除（TAA组合-MSIC AC World）
# 添加数值标签
for bar in bars3:
    width = bar.get_width()
    ax3.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:+.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=24)

# 设置x轴范围和标签字体大小
max_weight = df['MSCI AC World权重'].max()
ax1.set_xlim(0, max_weight * 1.1)
ax2.set_xlim(0, max_weight * 1.1)
max_diff = max(abs(df['差异'].min()), abs(df['差异'].max()))
ax3.set_xlim(-max_diff * 1.1, max_diff * 1.1)

# 设置轴标签字体大小
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', which='major', labelsize=24)

# 添加网格线
ax1.grid(True, axis='x', linestyle='--', alpha=0.3)
ax2.grid(True, axis='x', linestyle='--', alpha=0.3)
ax3.grid(True, axis='x', linestyle='--', alpha=0.3)

# 调整布局
plt.subplots_adjust(hspace=0.7)  # 增加子图间距以适应新的高度比例
plt.tight_layout(pad=3.0)  # 增加整体边距

# 保存图表
plt.savefig('TAA2_analysis.png', dpi=300, bbox_inches='tight')
print("图表已保存为 TAA2_analysis.png")
