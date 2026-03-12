#include "count.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task2 <n>\n";
        return 1;
    }

    const size_t n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (n == 0) {
        std::cerr << "n must be positive\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 500);

    thrust::host_vector<int> h_in(n);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    thrust::device_vector<int> d_in = h_in;
    thrust::device_vector<int> values;
    thrust::device_vector<int> counts;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    count(d_in, values, counts);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << values.back() << '\n';
    std::cout << counts.back() << '\n';
    std::cout << elapsed_ms << '\n';
    return 0;
}
