#include "matmul.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task4 <n> <threads_per_block>\n";
        return 1;
    }

    const size_t n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    const unsigned int threads_per_block = static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));
    if (n == 0 || threads_per_block == 0 || threads_per_block > MAX_THREADS_PER_BLOCK) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<half> h_A(n * n);
    std::vector<half> h_B(n * n);
    for (size_t i = 0; i < n * n; ++i) {
        h_A[i] = __float2half(dist(rng));
        h_B[i] = __float2half(dist(rng));
    }

    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_A), n * n * sizeof(half));
    cudaMalloc(reinterpret_cast<void**>(&d_B), n * n * sizeof(half));
    cudaMalloc(reinterpret_cast<void**>(&d_C), n * n * sizeof(half));

    cudaMemcpy(d_A, h_A.data(), n * n * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * n * sizeof(half), cudaMemcpyHostToDevice);

    auto run_and_print = [&](auto fn) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        fn();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        half last = __float2half(0.0f);
        cudaMemcpy(&last, d_C + (n * n - 1), sizeof(half), cudaMemcpyDeviceToHost);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << elapsed_ms << '\n';
        std::cout << __half2float(last) << '\n';
    };

    run_and_print([&]() { mmul_cuda(d_A, d_B, d_C, n, threads_per_block); });
    run_and_print([&]() { mmul_wmma(d_A, d_B, d_C, n, threads_per_block); });
    run_and_print([&]() { mmul_cublas(d_A, d_B, d_C, n); });

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
