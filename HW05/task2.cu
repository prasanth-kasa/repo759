#include <iostream>
#include <vector>
#include <cstdlib>
#include "reduce.cuh"

float rand_float() {
    return -1.0f + 2.0f * (static_cast<float>(std::rand()) / RAND_MAX);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 N threads_per_block\n";
        return 1;
    }
    
    unsigned int N = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);
    
    std::vector<float> h_in(N);
    for(unsigned int i = 0; i < N; ++i) h_in[i] = rand_float();
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    
    unsigned int initial_blocks = (N + (threads_per_block * 2) - 1) / (threads_per_block * 2);
    cudaMalloc(&d_out, initial_blocks * sizeof(float));
    
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    reduce(&d_in, &d_out, N, threads_per_block);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    float result = 0.0f;
    cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << result << "\n";
    std::cout << ms << "\n";
    
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}