#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include <cstdlib>
#include <iostream>
#include <random>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./task1_thrust <n>\n";
        return 1;
    }

    const size_t n = static_cast<size_t>(std::strtoull(argv[1], nullptr, 10));
    if (n == 0) {
        std::cerr << "n must be positive\n";
        return 1;
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    thrust::host_vector<float> h_in(n);
    for (size_t i = 0; i < n; ++i) {
        h_in[i] = dist(rng);
    }

    thrust::device_vector<float> d_in = h_in;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float result = thrust::reduce(d_in.begin(), d_in.end(), 0.0f, thrust::plus<float>());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << result << '\n';
    std::cout << elapsed_ms << '\n';
    return 0;
}
