#include <iostream>
#include <windows.h>
#include <vector>
#include <random>
#include <iomanip>
#include <xmmintrin.h>

using namespace std;

// 跨平台预取宏
#if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
#elif defined(_MSC_VER)
    #define PREFETCH_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
#else
    #define PREFETCH_NTA(addr)
#endif

// 最大规模设定为 3000万（约 240MB 数据，足以击穿目前所有桌面级 CPU 的 L3 Cache）
const int MAX_N = 30000000;
const int PREFETCH_DIST = 64; // 采用你上一轮跑出的最优甜点距离 D=64

double* A;
int* indices;
volatile double global_sink = 0.0;

void allocate_memory() {
    A = new double[MAX_N + PREFETCH_DIST];
    indices = new int[MAX_N + PREFETCH_DIST];
    for (int i = 0; i < MAX_N + PREFETCH_DIST; i++) {
        A[i] = 1.0; 
    }
}

int main() {
    allocate_memory();
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    mt19937 rng(42);

    // 打印 CSV 表头
    cout << "N,Repeats,Baseline(ms),Prefetch_D64(ms),Speedup" << endl;

    // 对数级步进扫描，精准捕捉跨越 L2/L3 Cache 到主存的物理边界
    int steps[] = {1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 30000000};

    for (int n : steps) {
        // 1. 动态限制当前工作集规模：生成 [0, n-1] 范围内的随机索引
        uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < n + PREFETCH_DIST; i++) {
            indices[i] = dist(rng);
        }

        // 2. 动态自适应重复次数（N 小时多跑几次抗误差，N 大时少跑防止等太久）
        int repeats = 30000000 / n;
        if (repeats < 1) repeats = 1;
        if (repeats > 5000) repeats = 5000;

        // ==========================================
        // 算法 A：无预取基准 (Baseline)
        // ==========================================
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < repeats; r++) {
            double sum_base = 0.0;
            for (int i = 0; i < n; i++) {
                sum_base += A[indices[i]];
            }
            global_sink = sum_base;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_base = ((tail - head) * 1000.0 / freq) / repeats;

        // ==========================================
        // 算法 B：最优距离 (D=64) 软件预取
        // ==========================================
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < repeats; r++) {
            double sum_pf = 0.0;
            for (int i = 0; i < n; i++) {
                // 向内存控制器提前 D=64 步发射非时间局部性预取指令
                PREFETCH_NTA(&A[indices[i + PREFETCH_DIST]]);
                sum_pf += A[indices[i]];
            }
            global_sink = sum_pf;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_pf = ((tail - head) * 1000.0 / freq) / repeats;

        // ==========================================
        // 性能统计
        // ==========================================
        double speedup = time_base / time_pf;

        cout << fixed << setprecision(5)
             << n << "," << repeats << ","
             << time_base << "," << time_pf << ","
             << speedup << endl;
    }

    delete[] A;
    delete[] indices;
    return 0;
}