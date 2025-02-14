#include <stdio.h>
#include <chrono>

typedef double real;

const int N = 100000000; // 数组长度

void __global__ reduce_global(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;  // 获取线程在块内的索引
    real *x = d_x + blockDim.x * blockIdx.x;  // 计算当前块处理的数据起始位置
    // real *x = &d_x[blockDim.x * blockIdx.x];  // 与上一行等价
    // 这样定义的 x 在不同的线程块中指向全局内存中不同的地址，使得我们可以在不同的线程块中对数组
    // d_x 中不同的部分进行归约操作

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    // 等价于 for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
    {
        if (tid < offset)
        {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[blockIdx.x] = x[0];
    }
}


void __global__ reduce_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;  // 获取线程在块内的索引
    const int bid = blockIdx.x;  // 获取块索引
    const int n = bid * blockDim.x + tid;  // 计算当前线程处理的数据索引
    __shared__ real s_y[128]; // 定义共享内存
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // 将全局内存中的数据拷贝到共享内存中
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}


void __global__ reduce_dynamic_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;  // 获取线程在块内的索引
    const int bid = blockIdx.x;  // 获取块索引
    const int n = bid * blockDim.x + tid;  // 计算当前线程处理的数据索引
    extern __shared__ real s_y[]; // 定义动态共享内存
    s_y[tid] = (n < N) ? d_x[n] : 0.0;  // 将全局内存中的数据拷贝到共享内存中
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            s_y[tid] += s_y[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}

// 执行归约操作
// reduce_dynamic_shared<<<grid_size, block_size, sizeof(real) * block_size>>>(d_x, d_y);