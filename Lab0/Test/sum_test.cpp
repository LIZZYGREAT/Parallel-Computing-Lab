#include <iostream>
#include <windows.h>
#include <iomanip>

using namespace std;

const int MAX_N = 100000000; 
double* A;

volatile double global_sink = 0.0;

void allocate_and_init() {
    A = new double[MAX_N];
    for (int i = 0; i < MAX_N; i++) {
        A[i] = (i % 10) * 0.1;
    }
}

// 递归归并算法
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

    cout << "N,Repeats,1-way(ms),2-way(ms),4-way(ms),Merge(ms),Speedup_2,Speedup_4" << endl;

    int steps[] = {1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000};

    for (int n : steps) {
        // 动态计算重复次数：保证每个规模的测试至少运行一定时间，消除微秒级误差
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

        // 输出 CSV 行
        cout << fixed << setprecision(5)
             << n << "," << repeats << ","
             << time1 << "," << time2 << "," << time4 << "," << time_merge << ","
             << sp2 << "," << sp4 << endl;
    }

    delete[] A;
    return 0;
}