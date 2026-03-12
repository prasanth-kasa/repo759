#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task1_cub <n>\n";
        return 1;
    }

    const size_t n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (n == 0) {
        std::cerr << "n must be positive\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float* h_in = new float[n];
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    float* d_in = nullptr;
    float* d_out = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cudaMalloc(reinterpret_cast<void**>(&d_in), n * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_out), sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    float result = 0.0f;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_temp_storage);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;

    std::cout << result << '\n';
    std::cout << elapsed_ms << '\n';
    return 0;
}
