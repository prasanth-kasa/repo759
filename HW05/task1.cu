#include <iostream>
#include <vector>
#include <cstdlib>
#include "matmul.cuh"

// --- Run INT ---
void run_matmul_int(unsigned int n, unsigned int block_dim) {
    size_t elements = n * n;
    std::vector<int> h_A(elements, 1);
    std::vector<int> h_B(elements, 2);
    std::vector<int> h_C(elements, 0);
    
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, elements * sizeof(int));
    cudaMalloc(&d_B, elements * sizeof(int));
    cudaMalloc(&d_C, elements * sizeof(int));
    
    cudaMemcpy(d_A, h_A.data(), elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), elements * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_1(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_C.data(), d_C, elements * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << h_C[0] << "\n";
    std::cout << h_C[elements - 1] << "\n";
    std::cout << ms << "\n";
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// --- Run FLOAT ---
void run_matmul_float(unsigned int n, unsigned int block_dim) {
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
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// --- Run DOUBLE ---
void run_matmul_double(unsigned int n, unsigned int block_dim) {
    size_t elements = n * n;
    std::vector<double> h_A(elements, 1.0);
    std::vector<double> h_B(elements, 2.0);
    std::vector<double> h_C(elements, 0.0);
    
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, elements * sizeof(double));
    cudaMalloc(&d_B, elements * sizeof(double));
    cudaMalloc(&d_C, elements * sizeof(double));
    
    cudaMemcpy(d_A, h_A.data(), elements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), elements * sizeof(double), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matmul_3(d_A, d_B, d_C, n, block_dim);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaMemcpy(h_C.data(), d_C, elements * sizeof(double), cudaMemcpyDeviceToHost);
    
    std::cout << h_C[0] << "\n";
    std::cout << h_C[elements - 1] << "\n";
    std::cout << ms << "\n";
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n block_dim\n";
        return 1;
    }
    
    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);
    
    run_matmul_int(n, block_dim);
    run_matmul_float(n, block_dim);
    run_matmul_double(n, block_dim);
    
    return 0;
}