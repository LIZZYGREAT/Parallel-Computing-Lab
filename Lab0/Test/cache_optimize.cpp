#include <iostream>
#include <windows.h>
#include <cmath>

using namespace std;

const int N = 8000; 

double A[N][N];
double x[N];
double y_trivial[N];
double y_optimized[N];

void init() {
    for (int i = 0; i < N; i++) {
        x[i] = i % 100;
        y_trivial[i] = 0.0;
        y_optimized[i] = 0.0;
        for (int j = 0; j < N; j++) {
            A[i][j] = i + j;
        }
    }
}

int main() {
    init();

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 算法 a) 逐列访问元素的平凡算法
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    
    // 平凡算法：外层控制列(j)，内层控制行(i)。
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            y_trivial[j] += A[i][j] * x[i];
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "time for no optimization: " 
         << (tail - head) * 1000.0 / freq << " ms" << endl;


    // 算法 b) Cache 优化算法
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    
    // 优化算法：外层控制行(i)，内层控制列(j)。
    for (int i = 0; i < N; i++) {
        double temp_x = x[i]; 
        for (int j = 0; j < N; j++) {
            y_optimized[j] += A[i][j] * temp_x;
        }
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "time for optimization: " 
         << (tail - head) * 1000.0 / freq << " ms" << endl;


    // 正确性校验
    bool is_correct = true;
    for (int i = 0; i < N; i++) {
        // 浮点数比较，允许极其微小的精度误差
        if (abs(y_trivial[i] - y_optimized[i]) > 1e-6) {
            is_correct = false;
            break;
        }
    }
    cout << endl << "correct? " << (is_correct ? "yes" : "no") << endl;

    return 0;
}