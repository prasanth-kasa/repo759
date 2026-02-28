#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }
    
    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);
    size_t elements = n * n;
    
    std::vector<float> h_A(elements, 1.0f);
    std::vector<float> h_B(elements, 2.0f);
    std::vector<float> h_C(elements, 0.0f);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, elements * sizeof(float));
    cudaMalloc(&d_B, elements * sizeof(float));
    cudaMalloc(&d_C, elements * sizeof(float));
    
    cudaMemcpy(d_A, h_A.data(), elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), elements * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_2(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_C.data(), d_C, elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << h_C[0] << "\n";
    std::cout << h_C[elements - 1] << "\n";
    std::cout << ms << "\n";
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}