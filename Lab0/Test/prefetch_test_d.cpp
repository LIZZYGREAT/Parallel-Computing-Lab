#include <iostream>
#include <windows.h>
#include <vector>
#include <random>
#include <iomanip>
#include <xmmintrin.h>

using namespace std;

#if defined(__GNUC__) || defined(__clang__)
    #define PREFETCH_NTA(addr) __builtin_prefetch((addr), 0, 0)
#elif defined(_MSC_VER)
    #define PREFETCH_NTA(addr) _mm_prefetch((const char*)(addr), _MM_HINT_NTA)
#else
    #define PREFETCH_NTA(addr)
#endif

const int FIXED_N = 30000000;
const int MAX_D = 1024; 

double* A;
int* indices;
volatile double global_sink = 0.0;

void allocate_memory() {
    A = new double[FIXED_N + MAX_D];
    indices = new int[FIXED_N + MAX_D];
    for (int i = 0; i < FIXED_N + MAX_D; i++) {
        A[i] = 1.0; 
    }
    
    // 生成真随机索引
    mt19937 rng(42);
    uniform_int_distribution<int> dist(0, FIXED_N - 1);
    for (int i = 0; i < FIXED_N + MAX_D; i++) {
        indices[i] = dist(rng);
    }
}

int main() {
    allocate_memory();
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    cout << "Distance_D,Repeats,Time(ms),Speedup" << endl;

    int repeats = 3;

    // 算法 A：无预取基准
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int r = 0; r < repeats; r++) {
        double sum_base = 0.0;
        for (int i = 0; i < FIXED_N; i++) {
            sum_base += A[indices[i]];
        }
        global_sink = sum_base;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    double time_base = ((tail - head) * 1000.0 / freq) / repeats;
    
    // 输出基准数据
    cout << fixed << setprecision(5) << 0 << "," << repeats << "," << time_base << "," << 1.00000 << endl;

    // 算法 B：梯度扫描预取距离 D
    int distances[] = {4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};

    for (int d : distances) {
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < repeats; r++) {
            double sum_pf = 0.0;
            for (int i = 0; i < FIXED_N; i++) {
                // 向内存控制器提前 D 步发射预取指令
                PREFETCH_NTA(&A[indices[i + d]]);
                sum_pf += A[indices[i]];
            }
            global_sink = sum_pf;
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_pf = ((tail - head) * 1000.0 / freq) / repeats;

        double speedup = time_base / time_pf;

        cout << fixed << setprecision(5)
             << d << "," << repeats << ","
             << time_pf << "," << speedup << endl;
    }

    delete[] A;
    delete[] indices;
    return 0;
}