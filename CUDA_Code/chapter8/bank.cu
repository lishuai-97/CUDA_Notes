#include <stdio.h>
#include <chrono>

typedef double real;
const int TILE_DIM = 32;  // C 风格

__global__ void transpose1(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];  // 定义共享内存
    int bx = blockIdx.x * TILE_DIM;;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    // 将一片矩阵数据从全局内存数组A中读出来，放在共享内存数组中
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();  // 线程块内的线程都执行完上面的操作后再继续往下执行，同步操作

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}