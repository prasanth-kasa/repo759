#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "matmul.cuh"

int main(int argc, char** argv) {
    if (argc != 3) return -1;
    
    size_t n = std::stoul(argv[1]);
    unsigned int threads_per_block = std::stoul(argv[2]);
    size_t total_elements = n * n;
    size_t size_bytes = total_elements * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[total_elements];
    float *h_B = new float[total_elements];
    float *h_C = new float[total_elements];
    
    // Fill with random numbers [-1, 1]
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < total_elements; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMalloc(&d_C, size_bytes);
    
    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice);
    
    // Setup CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul(d_A, d_B, d_C, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back and print
    cudaMemcpy(h_C, d_C, size_bytes, cudaMemcpyDeviceToHost);
    std::cout << h_C[total_elements - 1] << std::endl;
    std::cout << milliseconds << std::endl;
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}