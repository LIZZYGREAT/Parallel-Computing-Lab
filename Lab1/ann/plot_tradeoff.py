import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_latency_recall_tradeoff(csv_path):
    # 1. 检查数据文件是否存在
    if not os.path.exists(csv_path):
        print(f"[Error] 找不到数据文件: {csv_path}")
        print("请检查路径，确保已经从服务器拉取了 files/ 文件夹。")
        return

    # 2. 读取 CSV 数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] 读取 CSV 失败: {e}")
        return

    # 清理列名中的空格（防止手误导致的问题）
    df.columns = df.columns.str.strip()

    # 预期包含的列：'Threads', 'Top_C', 'Recall@10', 'Latency(us)'
    required_cols = {'Threads', 'Top_C', 'Recall@10', 'Latency(us)'}
    if not required_cols.issubset(df.columns):
        print(f"[Error] CSV 列名不匹配。找到的列: {df.columns.tolist()}")
        return

    # 3. 初始化学术风格图表配置
    plt.figure(figsize=(10, 7), dpi=150)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 获取所有的线程配置，用于循环画图
    threads = sorted(df['Threads'].unique())
    
    # 预设一些清晰的标记样式和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 4. 按线程数分组绘制曲线
    for idx, thread_count in enumerate(threads):
        # 筛选出当前线程的数据，并按照 Recall 排序（保证连线顺滑）
        subset = df[df['Threads'] == thread_count].sort_values(by='Recall@10')
        
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        plt.plot(subset['Recall@10'], subset['Latency(us)'], 
                 marker=marker, 
                 color=color, 
                 linewidth=2, 
                 markersize=8,
                 label=f'Threads = {thread_count}')
        
        # 5. 在关键节点上标注 Top_C 的值
        for _, row in subset.iterrows():
            plt.annotate(f"C={int(row['Top_C'])}", 
                         (row['Recall@10'], row['Latency(us)']),
                         textcoords="offset points", 
                         xytext=(0, 10), 
                         ha='center',
                         fontsize=9,
                         color='dimgray')

    # 6. 设置坐标轴与网格
    plt.xlabel('Recall@10', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (us)', fontsize=12, fontweight='bold')
    plt.title('ANN Search: Latency vs. Recall Trade-off', fontsize=14, fontweight='bold', pad=15)
    
    # 开启对数坐标轴（可选）——如果你的 Latency 跨度非常大，可以取消下面这行的注释
    # plt.yscale('log')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    
    # 7. 调整布局并保存
    plt.tight_layout()
    output_filename = 'latency_recall_tradeoff_curve.png'
    plt.savefig(output_filename, format='png')
    print(f"\n[Success] 图表已成功绘制并保存为当前目录下的 {output_filename}")
    
    # 也可以同时在窗口中显示
    # plt.show()

if __name__ == "__main__":
    # 指定你拉取下来的 CSV 文件路径
    data_file = "files/latency_recall_tradeoff.csv"
    plot_latency_recall_tradeoff(data_file)