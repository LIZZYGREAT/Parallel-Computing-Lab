import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取生成的 CSV 文件
data_file = '.\Lab0\Test\indirect_data.csv'

if not os.path.exists(data_file):
    print(f"错误：找不到 '{data_file}'，请先运行 C++ 程序生成数据。")
    exit()

# 如果遇到 0xff 报错，请加 encoding='utf-16'，若无报错则正常读取
try:
    df = pd.read_csv(data_file)
except UnicodeDecodeError:
    df = pd.read_csv(data_file, encoding='utf-16')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ==========================================
# 子图 1：绝对执行时间对比 (双对数坐标)
# ==========================================
ax1.plot(df['N'], df['Baseline(ms)'], marker='o', label='Baseline (No Prefetch)', color='#e74c3c', linewidth=2)
ax1.plot(df['N'], df['Prefetch_D64(ms)'], marker='D', label='Software Prefetch (D=64)', color='#1abc9c', linewidth=2)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Execution Time vs Data Size (Indirect Access)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Data Size N (Log Scale)', fontsize=12)
ax1.set_ylabel('Time (ms, Log Scale)', fontsize=12)
ax1.grid(True, which="both", ls="--", alpha=0.5)
ax1.legend()

# ==========================================
# 子图 2：软件预取加速比 (揭示 Cache 到主存的物理边界)
# ==========================================
ax2.plot(df['N'], df['Speedup'], marker='^', color='#3498db', linewidth=2)

ax2.set_xscale('log')
ax2.set_title('Prefetch Speedup vs Data Size', fontsize=13, fontweight='bold')
ax2.set_xlabel('Data Size N (Log Scale)', fontsize=12)
ax2.set_ylabel('Speedup (Baseline / Prefetch)', fontsize=12)

# 基准线
ax2.axhline(y=1.0, color='black', linestyle='--', label='Baseline (1x Speedup)')

# 标注 L3 Cache 的大致溢出区域 (视 CPU 而定，一般在 10^6 ~ 10^7 之间)
ax2.axvspan(10**6, 10**7, color='gray', alpha=0.1, label='Typical L3 Cache Boundary')

ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

plt.tight_layout()
output_img = 'indirect_prefetch_scale_plot.png'
plt.savefig(output_img, dpi=300)
print(f"绘图完成，已保存为 {output_img}")
plt.show()