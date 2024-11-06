#include "./include/foo.cuh"
#include <iostream>
#define N 512
int main()
{
    int h_a[N];
    int h_b[N];
    int h_c[N];
 
    for (int i = 0; i < N; ++i)
        h_a[i] = h_b[i] = i;
 
    h_vec_add(h_a, h_b, h_c, N);
 
    for (int i = 0; i < N; ++i)
        std::cout << h_c[i] << '\t';
    std::cout << '\n';
}
