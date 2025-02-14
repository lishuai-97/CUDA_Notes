#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

typedef float real;
const int TILE_DIM = 32;  // C++ 风格

__global__ void copy(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}

__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]);
    }
}

void printMatrix(const real *matrix, const int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    const int N = 1024; // 矩阵大小
    real *h_A, *h_B, *h_C;
    real *d_A, *d_B, *d_C;

    // 分配主机内存
    h_A = (real*)malloc(N * N * sizeof(real));
    h_B = (real*)malloc(N * N * sizeof(real));
    h_C = (real*)malloc(N * N * sizeof(real));

    // 初始化矩阵A
    for (int i = 0; i < N * N; i++) {
        h_A[i] = static_cast<real>(i);
    }

    // 分配设备内存
    cudaMalloc((void**)&d_A, N * N * sizeof(real));
    cudaMalloc((void**)&d_B, N * N * sizeof(real));
    cudaMalloc((void**)&d_C, N * N * sizeof(real));

    // 复制主机内存到设备内存
    cudaMemcpy(d_A, h_A, N * N * sizeof(real), cudaMemcpyHostToDevice);

    const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_size_y = grid_size_x;
    const dim3 block_size(TILE_DIM, TILE_DIM);
    const dim3 grid_size(grid_size_x, grid_size_y);

    // 复制矩阵
    copy<<<grid_size, block_size>>>(d_A, d_B, N);
    cudaMemcpy(h_B, d_B, N * N * sizeof(real), cudaMemcpyDeviceToHost);

    // 输出原始矩阵和复制的矩阵
    // printf("Original Matrix A:\n");
    // printMatrix(h_A, N);
    // printf("Copied Matrix B:\n");
    // printMatrix(h_B, N);

    // 转置矩阵方法1
    auto start1 = std::chrono::high_resolution_clock::now();
    transpose1<<<grid_size, block_size>>>(d_A, d_B, N);
    cudaDeviceSynchronize();
    auto end1 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_B, d_B, N * N * sizeof(real), cudaMemcpyDeviceToHost);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

    // 输出转置后的矩阵
    // printf("Transposed Matrix B (Method 1):\n");
    // printMatrix(h_B, N);
    printf("Time taken by transpose1: %lld microseconds\n", duration1);

    // 转置矩阵方法2
    auto start2 = std::chrono::high_resolution_clock::now();
    transpose2<<<grid_size, block_size>>>(d_A, d_C, N);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_C, d_C, N * N * sizeof(real), cudaMemcpyDeviceToHost);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

    // 输出转置后的矩阵
    // printf("Transposed Matrix C (Method 2):\n");
    // printMatrix(h_C, N);
    printf("Time taken by transpose2: %lld microseconds\n", duration2);

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}