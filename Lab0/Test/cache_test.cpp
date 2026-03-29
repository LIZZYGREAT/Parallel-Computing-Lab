#include <iostream>
#include <windows.h>
#include <iomanip>

using namespace std;

const int MAX_N = 8000;

double* A;
double* x;
double* y_trivial;
double* y_optimized;

void allocate_and_init() {
    A = new double[MAX_N * MAX_N];
    x = new double[MAX_N];
    y_trivial = new double[MAX_N];
    y_optimized = new double[MAX_N];

    for (int i = 0; i < MAX_N; i++) {
        x[i] = i % 100;
        y_trivial[i] = 0.0;
        y_optimized[i] = 0.0;
        for (int j = 0; j < MAX_N; j++) {
            A[i * MAX_N + j] = i + j;
        }
    }
}

int main() {
    allocate_and_init();

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    cout << "N,Repeats,Trivial_Time(ms),Optimized_Time(ms),Speedup" << endl;

    for (int n = 10; n <= MAX_N; n += (n < 100 ? 10 : (n < 1000 ? 100 : 1000))) {
        
        int repeats = 1;
        if (n <= 100) repeats = 10000;
        else if (n <= 1000) repeats = 100;
        else repeats = 5;

        // 测试平凡算法
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < repeats; r++) {
            // 每次测试前清空结果向量
            for (int i = 0; i < n; i++) y_trivial[i] = 0.0; 
            
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < n; i++) {
                    y_trivial[j] += A[i * MAX_N + j] * x[i];
                }
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_trivial = ((tail - head) * 1000.0 / freq) / repeats;

        // 测试 Cache 优化算法
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        for (int r = 0; r < repeats; r++) {
            for (int i = 0; i < n; i++) y_optimized[i] = 0.0;

            for (int i = 0; i < n; i++) {
                double temp_x = x[i];
                for (int j = 0; j < n; j++) {
                    y_optimized[j] += A[i * MAX_N + j] * temp_x;
                }
            }
        }
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        double time_optimized = ((tail - head) * 1000.0 / freq) / repeats;

        // 计算加速比
        double speedup = time_trivial / time_optimized;

        // 输出为 CSV 格式
        cout << fixed << setprecision(4) 
             << n << "," 
             << repeats << "," 
             << time_trivial << "," 
             << time_optimized << "," 
             << speedup << endl;
    }

    delete[] A;
    delete[] x;
    delete[] y_trivial;
    delete[] y_optimized;

    return 0;
}