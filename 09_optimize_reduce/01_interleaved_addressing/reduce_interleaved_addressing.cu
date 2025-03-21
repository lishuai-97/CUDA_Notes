#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int len = 32 * 1024 * 1024;

template <int BLOCKSIZE>
__global__ void reduce_naive_kernel(int *arr, int *out, int len)
{
    __shared__ int sdata[BLOCKSIZE];
    int tid = threadIdx.x;    // 线程 id (block 内)
    int bid = blockIdx.x;     // block id (grid 内)
    int bdim = blockDim.x;    // block 大小
    int i = bid * bdim + tid; // 全局 id

    // 将数据拷贝到共享内存
    if (i < len)
    {
        sdata[tid] = arr[i];
    }

    __syncthreads(); // 等待所有线程完成

    // 使用交错寻址
    for (int s = 1; s < bdim; s *= 2)
    {   
        // 计算当前线程应该处理的元素索引
        // 每轮迭代中，活跃线程数减半，但每个线程处理的元素位置改变
        int index = 2 * s * tid;
        // index + s < bdim 确保在共享内存范围内
        // bdim * bid + s < len 确保在全局内存范围内
        if ((index + s < bdim) && (bdim * bid + s < len))
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // 每个 block 的第一个线程将结果写入到 out 中
    if (tid == 0)
    {
        out[bid] = sdata[0];
    }
}

int main()
{
    int *arr = new int[len];
    int *out = new int[len];
    int *d_arr, *d_out;

    // 初始化数组
    for (int i = 0; i < len; i++)
    {
        arr[i] = 1;
    }

    // 分配内存
    cudaMalloc((void **)&d_arr, sizeof(int) * len);
    cudaMalloc((void **)&d_out, sizeof(int) * len);

    // 拷贝数据到显存
    cudaMemcpy(d_arr, arr, sizeof(int) * len, cudaMemcpyHostToDevice);

    // 计算 block 和 grid 的大小
    const int blocksize = 256;
    const int gridsize = (len + blocksize - 1) / blocksize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaEventQuery(start);

    // 调用 kernel 函数
    reduce_naive_kernel<blocksize><<<gridsize, blocksize>>>(d_arr, d_out, len);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time: %f ms.\n", elapsed_time);
    printf("elapsed time: %f us.\n", elapsed_time * 1000);

    // 拷贝数据到内存
    cudaMemcpy(out, d_out, sizeof(int) * len, cudaMemcpyDeviceToHost);

    // 计算结果
    long long sum = 0;
    for (int i = 0; i < gridsize; i++)
    {
        sum += out[i];
    }
    printf("sum = %d\n", sum);

    // 核对结果
    long long sum2 = 0;
    for (int i = 0; i < len; i++)
    {
        sum2 += arr[i];
    }

    if (sum == sum2)
    {
        printf("success\n");
    }
    else
    {
        printf("failed, the result is %d\n", sum2);
    }

    // 释放内存
    cudaFree(d_arr);
    cudaFree(d_out);
    delete[] arr;
    delete[] out;
    return 0;
}