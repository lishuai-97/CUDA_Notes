## Windows环境下CUDA编程环境的搭建

Windows 环境配置同样需要安装 CUDA Toolkit，下载地址为：https://developer.nvidia.com/cuda-downloads。

在 Windows 进行安装时需要选自定义模式，采用精简模式安装后无法运行 nvcc 命令。

安装成功后可尝试 `nvcc --version` 检测下，编译时如果缺少 cl.exe，则需要安装 Microsoft Visual Studio(使用 C++的桌面开发版本即可)。安装完成后，将 cl.exe 所在路径添加到系统变量中，cl.exe 通常所在文件夹为`C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\{version}\bin\Hostx64\x64`，具体路径根据实际安装情况有所不同。

和 Linux 不同之处在于，安装 Toolkit 之后还需要配置下环境变量。默认系统会已经有 `CUDA_PATH` 和 `CUDA_PATH_V11.0`（11.0 应该是版本号），需要自己在额外添加如下环境变量：

```bash
CUDA_BIN_PATH: %CUDA_PATH%\bin
CUDA_LIB_PATH: %CUDA_PATH%\lib\x64
CUDA_SDK_PATH: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.6  #<---- 注意版本号可能不一样
CUDA_SDK_BIN_PATH: %CUDA_SDK_PATH%\bin\win64
CUDA_SDK_LIB_PATH: %CUDA_SDK_PATH%\common\lib\x64
```

- **注意**：CUDA在11.6版本以后不会将Samples集成在toolkit安装包中，因此在安装完CUDA 11.8以后会找不到对应的CUDA Samples目录。需要从GitHub上下载CUDA Samples，下载地址为：https://github.com/NVIDIA/cuda-samples （中间遇到的问题可以参考贴子[[1](https://forums.developer.nvidia.com/t/cuda-samples-build-time-error-windows-10-and-cuda-10-1-toolikt/71549)][[2](https://blog.csdn.net/zaizhipo936/article/details/140287171?spm=1001.2014.3001.5506)]解决！）

此外，还需要在系统变量 PATH 中添加如下变量：

```bash
%CUDA_BIN_PATH%
%CUDA_LIB_PATH%
%CUDA_SDK_BIN_PATH%
%CUDA_SDK_LIB_PATH%
```

最终，可以运行安装目录下 Nvidia 提供的测试 `.exe` 执行文件：`deviceQuery.exe、bandwidthTest.exe`，如果运行没有问题，则表示环境配置成功了.(在安装路径 `extras/demo_suite`目录里)

### 运行 Demo 样例

新建一个 `hello_world.cu` 文件（见此目录）:
```cpp
#include <stdio.h>

__global__ void cuda_say_hello(){
    printf("Hello world, CUDA! %d\n", threadIdx.x);
}

int main(){
    printf("Hello world, CPU\n");
    cuda_say_hello<<<1,1>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}
```

首先使用如下命令编译 `nvcc hello_world.cu -o hello_world`, 然后执行 `.\hello_world`, 会得到如下输出：

```bash
Hello world, CPU
Hello world, CUDA! 0
```