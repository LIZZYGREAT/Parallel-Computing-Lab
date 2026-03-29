import pandas as pd
import matplotlib.pyplot as plt
import os

data_file = '.\Lab0\Test\cache_data_2.csv'



# 读取CSV数据
df = pd.read_csv(data_file)

# 创建一个 1x2 的宽幅画布，适合插入到实验报告中
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 子图 1：执行时间对比 (绝对性能)
# ==========================================
# 使用圆点和方块标记数据点，区分颜色，符合学术制图规范
ax1.plot(df['N'], df['Trivial_Time(ms)'], marker='o', label='Trivial Algorithm (Col-Major)', color='#e74c3c', linewidth=2)
ax1.plot(df['N'], df['Optimized_Time(ms)'], marker='s', label='Optimized Algorithm (Row-Major)', color='#2ecc71', linewidth=2)

ax1.set_title('Execution Time vs Matrix Size (N)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Matrix Size (N)', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=12)

# ==========================================
# 子图 2：加速比曲线 (揭示 Cache 命中率跃变)
# ==========================================
ax2.plot(df['N'], df['Speedup'], marker='^', color='#3498db', linewidth=2)

ax2.set_title('Speedup Ratio vs Matrix Size (N)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Matrix Size (N)', fontsize=12)
ax2.set_ylabel('Speedup (Trivial Time / Optimized Time)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# 添加一条 y=1 的黑色虚线作为加速比基准线 (1x)
ax2.axhline(y=1.0, color='black', linestyle='--', label='Baseline (1x Speedup)')
ax2.legend(fontsize=12)

# 调整子图间距，防止文字重叠
plt.tight_layout()

# 导出为高分辨率图片，直接用于 Word/LaTeX 实验报告
output_img = 'cache_2_analysis_plot.png'
plt.savefig(output_img, dpi=300)
print(f"绘图成功！高分辨率图表已保存为当前目录下的: {output_img}")

# 在运行脚本时同时展示窗口
plt.show()