#include <iostream>
#include <windows.h>
#include <cmath>

using namespace std;

const int N = 100000000; 
double A[N];

void init() {
    for (int i = 0; i < N; i++) {
        A[i] = (i % 10) * 0.1; 
    }
}

double merge(const double A[N],int start,int end){
    if(start>end) return 0;
    if(start==end){
         return A[start];
    }
    int mid = start+(end-start)/2;
    return merge(A,start,mid)+merge(A,mid+1,end);
}

int main() {
    init();

    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    // 算法 a) 逐个累加的平凡算法
    double sum_trivial = 0.0;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    
    for (int i = 0; i < N; i++) {
        sum_trivial += A[i];
    }

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "time for one-way:" 
         << (tail - head) * 1000.0 / freq << " ms, result: " << sum_trivial << endl;



    // 算法 b-1) 两路链式累加
    double sum_ilp2_1 = 0.0;
    double sum_ilp2_2 = 0.0;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    
    for (int i = 0; i < N; i += 2) {
        sum_ilp2_1 += A[i];
        sum_ilp2_2 += A[i+1];
    }
    double sum_ilp2 = sum_ilp2_1 + sum_ilp2_2; // 循环外合并

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "time for two-way:" 
         << (tail - head) * 1000.0 / freq << " ms, result: " << sum_ilp2 << endl;


    // 算法 b-2) 四路链式累加
    double sum_ilp4_1 = 0.0;
    double sum_ilp4_2 = 0.0;
    double sum_ilp4_3 = 0.0;
    double sum_ilp4_4 = 0.0;
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    
    for (int i = 0; i < N; i += 4) {
        sum_ilp4_1 += A[i];
        sum_ilp4_2 += A[i+1];
        sum_ilp4_3 += A[i+2];
        sum_ilp4_4 += A[i+3];
    }
    double sum_ilp4 = sum_ilp4_1 + sum_ilp4_2 + sum_ilp4_3 + sum_ilp4_4;

    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "time for four-way:" 
         << (tail - head) * 1000.0 / freq << " ms, results: " << sum_ilp4 << endl;

    //算法 c) 递归归并求和
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    
    double sum_merge = merge(A,0,N-1);


    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "time for merge:" 
         << (tail - head) * 1000.0 / freq << " ms, results: " << sum_merge << endl;

    return 0;
}