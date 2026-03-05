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
    cudaMallocManaged(&input, n * sizeof(float)); // [cite: 45]
    cudaMallocManaged(&output, n * sizeof(float));

    // Fill with random numbers [-1, 1] [cite: 45]
    for (unsigned int i = 0; i < n; ++i) {
        input[i] = -1.0f + 2.0f * (static_cast<float>(rand()) / RAND_MAX);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    scan(input, output, n, threads_per_block); // [cite: 46]
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the last element of the output [cite: 47]
    std::cout << output[n - 1] << std::endl;
    // Print time taken [cite: 48]
    std::cout << milliseconds << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(input);
    cudaFree(output);

    return 0;
}