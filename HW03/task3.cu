#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n> [threads_per_block]" << std::endl;
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    
    int threads_per_block = 512;
    if (argc >= 3) {
        threads_per_block = std::atoi(argv[2]);
    }

    size_t bytes = n * sizeof(float);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_a(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_b(0.0f, 1.0f);

    std::vector<float> hA(n);
    std::vector<float> hB(n);

    for(unsigned int i = 0; i < n; i++) {
        hA[i] = dist_a(gen);
        hB[i] = dist_b(gen);
    }

    float *dA, *dB;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);

    cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEventRecord(start);
    vscale<<<num_blocks, threads_per_block>>>(dA, dB, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hB.data(), dB, bytes, cudaMemcpyDeviceToHost);

    std::cout << ms << std::endl;
    std::cout << hB[0] << std::endl;
    std::cout << hB[n-1] << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);

    return 0;
}