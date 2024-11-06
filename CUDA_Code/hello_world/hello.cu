#include <stdio.h>

__global__ void hello_world(void) {
    printf("GPU: Hello World!\n");
}


int main(void) {

    printf("CPU: Hello World!\n");

    hello_world<<<1, 10>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceReset();

    return 0;

}