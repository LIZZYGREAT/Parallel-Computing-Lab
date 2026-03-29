import pandas as pd
import matplotlib.pyplot as plt
import os

data_file = '.\Lab0\Test\sum_data_0.csv'


df = pd.read_csv(data_file)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ==========================================
# 子图 1：绝对执行时间 (使用双对数坐标 log-log)
# ==========================================
ax1.plot(df['N'], df['1-way(ms)'], marker='o', label='Trivial (1-way)', color='#7f8c8d', linewidth=2)
ax1.plot(df['N'], df['2-way(ms)'], marker='s', label='ILP Optimized (2-way)', color='#f39c12', linewidth=2)
ax1.plot(df['N'], df['4-way(ms)'], marker='^', label='ILP Optimized (4-way)', color='#e74c3c', linewidth=2)
ax1.plot(df['N'], df['Merge(ms)'], marker='x', label='Recursive Merge', color='#9b59b6', linewidth=2, linestyle=':')

ax1.set_xscale('log') # 关键：X轴使用对数系
ax1.set_yscale('log') # 关键：Y轴时间跨度太大，使用对数系
ax1.set_title('Execution Time vs Array Size (Log-Log Scale)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Array Size N (Log Scale)', fontsize=12)
ax1.set_ylabel('Time (ms, Log Scale)', fontsize=12)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ==========================================
# 子图 2：超标量加速比 (X轴对数，Y轴线性)
# ==========================================
ax2.plot(df['N'], df['Speedup_2'], marker='s', label='2-way Speedup', color='#f39c12', linewidth=2)
ax2.plot(df['N'], df['Speedup_4'], marker='^', label='4-way Speedup', color='#e74c3c', linewidth=2)

ax2.set_xscale('log')
ax2.set_title('ILP Speedup vs Array Size (Log-Linear Scale)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Array Size N (Log Scale)', fontsize=12)
ax2.set_ylabel('Speedup Ratio', fontsize=12)

# 基准线
ax2.axhline(y=1.0, color='black', linestyle='--', label='Baseline (1x)')
ax2.axhline(y=2.0, color='gray', linestyle=':', label='Theoretical Max (2x)')
ax2.axhline(y=4.0, color='gray', linestyle=':', label='Theoretical Max (4x)')

ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig('ilp_analysis_plot.png', dpi=300)
print("绘图完成，已保存为 ilp_analysis_plot.png")
plt.show()