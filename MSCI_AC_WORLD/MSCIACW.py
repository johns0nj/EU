import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建全球权重数据
global_weights = {
    '国家': ['美国', '欧洲', '日本', '中国', '加拿大', '其他国家'],
    '权重（%）': [65.75, 16.8, 4.71, 3.01, 2.71, 6.02]
}

# 创建欧洲权重数据
europe_weights = {
    '国家': ['英国', '法国', '瑞士', '德国', '荷兰', '其他欧洲国家'],
    '权重（%）': [3.27, 2.73, 2.47, 2.31, 1.11, 0]  # 先设为0，后面计算
}

# 计算其他欧洲国家的权重
total_europe_weight = 16.8
current_europe_weight = sum(europe_weights['权重（%）'][:-1])
europe_weights['权重（%）'][-1] = total_europe_weight - current_europe_weight

# 转换为DataFrame
global_df = pd.DataFrame(global_weights)
europe_df = pd.DataFrame(europe_weights)

# 保存到Excel文件
with pd.ExcelWriter('MSCIACW.xlsx') as writer:
    global_df.to_excel(writer, sheet_name='全球权重', index=False)
    europe_df.to_excel(writer, sheet_name='欧洲权重', index=False)

print("数据已成功保存至 MSCIACW.xlsx")

# 读取Excel文件
with pd.ExcelFile('MSCIACW.xlsx') as xls:
    global_df = pd.read_excel(xls, sheet_name='全球权重')
    europe_df = pd.read_excel(xls, sheet_name='欧洲权重')

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# 全球权重饼图
global_df.plot.pie(y='权重（%）', labels=global_df['国家'], autopct='%1.1f%%',
                   startangle=90, ax=ax1, legend=False, textprops={'fontsize': 20})
ax1.set_title('全球权重分布', fontsize=24)
ax1.set_ylabel('')

# 欧洲权重饼图
europe_df.plot.pie(y='权重（%）', labels=europe_df['国家'], autopct='%1.1f%%',
                   startangle=90, ax=ax2, legend=False, textprops={'fontsize': 20})
ax2.set_title('欧洲权重分布', fontsize=24)
ax2.set_ylabel('')

# 调整布局
plt.tight_layout()

# 将图表保存到Excel文件
max_retries = 3
retry_delay = 1  # 重试间隔时间（秒）

for attempt in range(max_retries):
    try:
        # 使用 openpyxl 引擎
        with pd.ExcelWriter('MSCIACW_with_charts.xlsx', engine='openpyxl') as writer:
            global_df.to_excel(writer, sheet_name='全球权重', index=False)
            europe_df.to_excel(writer, sheet_name='欧洲权重', index=False)
            
            # 将图表保存为图片
            plt.savefig('pie_charts.png', dpi=300, bbox_inches='tight')
            
            # 获取工作表对象
            workbook = writer.book
            worksheet = writer.sheets['全球权重']
            
            # 插入图片
            from openpyxl.drawing.image import Image
            img = Image('pie_charts.png')
            worksheet.add_image(img, 'D2')
            
        break  # 如果成功，退出循环
    except PermissionError:
        if attempt < max_retries - 1:  # 如果不是最后一次尝试
            time.sleep(retry_delay)  # 等待一段时间后重试
        else:
            print("错误：无法保存文件，请确保文件未被其他程序打开")
            raise  # 最后一次尝试失败，抛出异常

print("数据和图表已成功保存至 MSCIACW_with_charts.xlsx") 