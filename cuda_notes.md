# CUDA 编程笔记

- [CUDA 编程笔记](#cuda-编程笔记)
    - [1. 典型的CUDA程序基本框架](#1-典型的cuda程序基本框架)
    - [2. CUDA编程规范](#2-cuda编程规范)
    - [3. 常用的CUDA计时函数](#3-常用的cuda计时函数)
    - [4. CUDA程序性能剖析](#4-cuda程序性能剖析)
    - [5. GPU加速的关键因素](#5-gpu加速的关键因素)


### 1. 典型的CUDA程序基本框架

```cpp
头文件包含
常量定义（或者宏定义）
C++ 自定义函数和 CUDA 核函数的声明（原型）

int main(void)
{
    分配主机与设备内存
    初始化主机中的数据
    将某些数据从主机复制到设备
    调用核函数在设备中进行计算
    将某些数据从设备复制到主机
    释放主机与设备内存
}

C++ 自定义函数和 CUDA 核函数的定义
```

### 2. CUDA编程规范

- 为了区分主机和设备中的变量，遵循CUDA编程的传统，用`d_`前缀表示设备变量，用`h_`前缀表示主机变量。

### 3. 常用的CUDA计时函数

```cpp
#include "error.cuh"

cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start);  // 此处不能用 CHECK 宏函数（见第 4 章的讨论）

需要计时的代码块 （主机代码、设备代码、混合代码）

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms.\n", elapsed_time);

CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
```

### 4. CUDA程序性能剖析

**(1) 通过`nvprof`工具进行性能剖析，可以查看程序的运行时间、内存使用情况、核函数的调用次数等信息**。（8.0算力以下可以用）

```bash
nvprof ./a.out
```

注：8.0以上算力显卡会报错：

```bash
E:\Code_Exe\CUDA_Notes\CUDA_Code\chapter5>nvprof .\add3memory.exe 
======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
                  Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
```

**(2) 使用`Nsight Systems`工具进行性能剖析**（8.0及以上算力）。

```bash
# nsys profile --stats=true -o report_name ./your_program
nsys profile --stats=true .\add3memory_dp.exe
```

- `profile` 是 Nsight Systems 的主要命令，表示进行性能分析。
- `--stats=true` 表示在分析后打印统计信息。
- `-o report_name` 表示生成的报告文件名。
- `./your_program` 是要分析的可执行程序。

这个命令会生成一个`.qdrep`格式的报告文件，它包含详细的性能数据。

注：使用过程中可能会遇到:
```bash
Unexpected exception thrown while launching the application.
Dynamic exception type: class std::range_error
std::exception::what: bad conversion
```
原因是`nsys`的版本过旧，和cuda11.8不匹配，下载安装[nsys2024.6.1](https://developer.nvidia.com/tools-downloads)，并添加环境变量(`C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.1\target-windows-x64`)+重启之后，运行代码，即可通过。

**(3) 使用`Nsight Compute`工具进行性能分析**

```bash
ncu -o profile_result .\add3memory.exe
```


### 5. GPU加速的关键因素

一个CUDA程序能够获得高性能的必要（但不充分）条件有如下几点：

* 数据传输比例较小；
* 核函数的算术强度较高；
* 核函数中定义的线程数目较多。

所以在编写和优化CUDA程序时，要做到以下几点：

* 减少主机与设备之间的数据传输；
* 提高核函数的算术强度；
* 增大核函数的并行规模。