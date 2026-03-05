#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "mmul.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n n_tests\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    int n_tests = std::atoi(argv[2]);

    size_t bytes = n * n * sizeof(float);
    float *A, *B, *C;

    // Allocate managed memory [cite: 24]
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    // Fill with random floats in [-1, 1] [cite: 24]
    for (int i = 0; i < n * n; ++i) {
        A[i] = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
        B[i] = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
        C[i] = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    // CUDA Events for timing 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < n_tests; ++i) {
        mmul(handle, A, B, C, n); // [cite: 25]
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print average time in ms 
    std::cout << milliseconds / n_tests << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}