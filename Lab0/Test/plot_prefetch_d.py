import pandas as pd
import matplotlib.pyplot as plt
import os

data_file = '.\Lab0\Test\indirect_d_data.csv'

if not os.path.exists(data_file):
    print(f"错误：找不到 '{data_file}'，请先运行 C++ 程序生成数据。")
    exit()

try:
    df = pd.read_csv(data_file)
except UnicodeDecodeError:
    df = pd.read_csv(data_file, encoding='utf-16')

# 提取基准时间 (D=0 时的行)
baseline_time = df[df['Distance_D'] == 0]['Time(ms)'].values[0]

# 过滤出有预取的数据点 (D > 0)
df_pf = df[df['Distance_D'] > 0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==========================================
# 子图 1：绝对执行时间随预取距离的变化
# ==========================================
ax1.plot(df_pf['Distance_D'], df_pf['Time(ms)'], marker='o', color='#3498db', linewidth=2, label='Prefetch Time')
ax1.axhline(y=baseline_time, color='#e74c3c', linestyle='--', label='Baseline Time (No Prefetch)')

ax1.set_title('Execution Time vs Prefetch Distance (D)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Prefetch Distance (D)', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# ==========================================
# 子图 2：加速比的“甜点”曲线
# ==========================================
ax2.plot(df_pf['Distance_D'], df_pf['Speedup'], marker='^', color='#2ecc71', linewidth=2, label='Prefetch Speedup')
ax2.axhline(y=1.0, color='black', linestyle='--', label='Baseline (1x Speedup)')

# 寻找最高加速比及其对应的 D 值
max_speedup = df_pf['Speedup'].max()
optimal_d = df_pf[df_pf['Speedup'] == max_speedup]['Distance_D'].values[0]

# 在图上标注 Sweet Spot
ax2.annotate(f'Sweet Spot\nD={optimal_d}\n{max_speedup:.2f}x', 
             xy=(optimal_d, max_speedup), 
             xytext=(optimal_d + 30, max_speedup - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=11, fontweight='bold', color='#27ae60')

ax2.set_title('Speedup Ratio vs Prefetch Distance (D)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Prefetch Distance (D)', fontsize=12)
ax2.set_ylabel('Speedup Ratio', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
output_img = 'prefetch_distance_tune_plot.png'
plt.savefig(output_img, dpi=300)
print(f"绘图完成，已保存为 {output_img}")
plt.show()