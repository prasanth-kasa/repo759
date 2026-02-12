#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void factorialKernel(int* dA) {
    int x = threadIdx.x;
    int val = x + 1;
    
    int result = 1;
    for (int i = 1; i <= val; i++) {
        result *= i;
    }
    
    dA[x] = result;
}

int main() {
    const int N = 8;
    const int bytes = N * sizeof(int);

    int* dA;
    cudaMalloc(&dA, bytes);

    factorialKernel<<<1, 8>>>(dA);
    cudaDeviceSynchronize();

    std::vector<int> hA(N);
    cudaMemcpy(hA.data(), dA, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << hA[i] << std::endl;
    }

    cudaFree(dA);
    return 0;
}