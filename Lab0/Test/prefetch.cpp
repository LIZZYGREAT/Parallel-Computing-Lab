#include <iostream>
#include <windows.h>
#include <iomanip>
#include <xmmintrin.h>

#if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
#elif defined(_MSC_VER)
    #define PREFETCH_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
#else
    #define PREFETCH_NTA(addr) // 若是不支持的编译器，则退化为空操作
#endif

using namespace std;

const int MAX_N = 100000000; 
double* A;

volatile double global_sink = 0.0;

void allocate_and_init() {
    // 严谨工程处理：尾部额外分配 256 个元素的 Padding 缓冲区
    // 保证在数组末尾调用预取指令 &A[i + 240] 时，物理内存地址合法，避免 C++ 越界 UB
    A = new double[MAX_N + 256];
    for (int i = 0; i < MAX_N + 256; i++) {
        A[i] = (i % 10) * 0.1;
    }
}

double merge_sum(const double arr[], int start, int end) {
    if (start > end) return 0;
    if (start == end) return arr[start];
    int mid = start + (end - start) / 2;
    return merge_sum(arr, start, mid) + merge_sum(arr, mid + 1, end);
}

int main() {
    allocate_and_init();
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    cout << "N,Repeats,1-way(ms),2-way(ms),4-way(ms),4-way+Prefetch(ms),Merge(ms),Speedup_2,Speedup_4,Speedup_4_PF" << endl;

    int steps[] = {1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000};

    for (int n : steps) {
        int repeats = 100000000 / n;
        if (repeats < 1) repeats = 1;
        if (repeats > 10000) repeats = 10000;

        // 算法 a: 平凡算法 (1路)
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int r = 0; r < repeats; r++) {
            double sum1 = 0;
            for(int i = 0; i < n; i++) sum1 += A[i];
            global_sink = sum1;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time1 = ((tail - head) * 1000.0 / freq) / repeats;

        // 算法 b-1: 2路超标量
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int r = 0; r < repeats; r++) {
            double s1 = 0, s2 = 0;
            for(int i = 0; i < n; i += 2) {
                s1 += A[i];
                s2 += A[i+1];
            }
            double sum2 = s1 + s2;
            global_sink = sum2;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time2 = ((tail - head) * 1000.0 / freq) / repeats;

        // 算法 b-2: 4路超标量
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int r = 0; r < repeats; r++) {
            double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
            for(int i = 0; i < n; i += 4) {
                s1 += A[i]; s2 += A[i+1]; 
                s3 += A[i+2]; s4 += A[i+3];
            }
            double sum4 = s1 + s2 + s3 + s4;
            global_sink = sum4;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time4 = ((tail - head) * 1000.0 / freq) / repeats;

        // 算法 b-3: 4路超标量 + 软件预取
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int r = 0; r < repeats; r++) {
            double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
            for(int i = 0; i < n; i += 4) {
                // 预取节流：利用位运算 (i & 7) == 0 保证每 8 个元素（即 1 个 Cache Line）仅发射一次预取指令
                if ((i & 7) == 0) {
                    // rw=0(读), locality=0(非时间局部性，避免污染 Cache)
                    PREFETCH_NTA(&A[i + 240]);
                }
                s1 += A[i]; s2 += A[i+1]; 
                s3 += A[i+2]; s4 += A[i+3];
            }
            double sum_pf = s1 + s2 + s3 + s4;
            global_sink = sum_pf;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_pf = ((tail - head) * 1000.0 / freq) / repeats;

        // 算法 c: 递归算法
        int merge_repeats = repeats / 10;
        if (merge_repeats < 1) merge_repeats = 1;
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for(int r = 0; r < merge_repeats; r++) {
            double sum_m = merge_sum(A, 0, n - 1);
            global_sink = sum_m;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_merge = ((tail - head) * 1000.0 / freq) / merge_repeats;

        // 计算加速比
        double sp2 = time1 / time2;
        double sp4 = time1 / time4;
        double sp4_pf = time1 / time_pf;

        // 输出 CSV 行
        cout << fixed << setprecision(5)
             << n << "," << repeats << ","
             << time1 << "," << time2 << "," << time4 << "," << time_pf << "," << time_merge << ","
             << sp2 << "," << sp4 << "," << sp4_pf << endl;
    }

    delete[] A;
    return 0;
}