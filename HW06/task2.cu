#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "scan.cuh"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n threads_per_block\n";
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    float *input, *output;
    cudaMallocManaged(&input, n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));

    for (unsigned int i = 0; i < n; ++i) {
        input[i] = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    scan(input, output, n, threads_per_block);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    scan(input, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << output[n - 1] << std::endl;
    
    std::cout << milliseconds << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);

    return 0;
}