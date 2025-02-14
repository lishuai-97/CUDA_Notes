#include <stdio.h>
#include <chrono>

typedef double real;

real reduce(const real *x, const int N)
{
    real sum = 0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}

int main() {
    const int N = 100000000; // 数组长度
    real *h_x = (real*)malloc(N * sizeof(real));

    // 初始化数组
    for (int i = 0; i < N; ++i) {
        h_x[i] = 1.23;
    }

    // 调用reduce函数并计时
    auto start = std::chrono::high_resolution_clock::now();
    real sum = reduce(h_x, N);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Sum: %f\n", sum);
    printf("Time taken by reduce: %lld milliseconds\n", duration);

    // 释放内存
    free(h_x);

    return 0;
}