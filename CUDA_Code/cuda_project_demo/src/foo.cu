#include "foo.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
 
//Kernel
__global__ void d_vec_add(int *d_a, int *d_b, int *d_c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_c[i] = d_a[i] + d_b[i];
}
 
void h_vec_add(int *a, int *b, int *c, int n)
{
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(int) * n);
    cudaMalloc((void **)&d_b, sizeof(int) * n);
    cudaMalloc((void **)&d_c, sizeof(int) * n);
 
    cudaMemcpy(d_a, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * n, cudaMemcpyHostToDevice);
 
    dim3 DimGrid(n / BX + 1, 1, 1);
    dim3 DimBlock(BX, 1, 1);
 
    d_vec_add<<<DimGrid, DimBlock>>>(d_a, d_b, d_c, n);
 
    cudaMemcpy(c, d_c, sizeof(int) * n, cudaMemcpyDeviceToHost);
 
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
 