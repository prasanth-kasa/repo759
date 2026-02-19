#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char** argv) {
    if (argc != 4) return -1;
    
    unsigned int n = std::stoul(argv[1]);
    unsigned int R = std::stoul(argv[2]);
    unsigned int threads_per_block = std::stoul(argv[3]);
    
    size_t img_size = n * sizeof(float);
    size_t mask_size = (2 * R + 1) * sizeof(float);
    
    float *h_image = new float[n];
    float *h_output = new float[n];
    float *h_mask = new float[2 * R + 1];
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (unsigned int i = 0; i < n; ++i) h_image[i] = dist(gen);
    for (unsigned int i = 0; i < 2 * R + 1; ++i) h_mask[i] = dist(gen);
    
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_mask, mask_size);
    cudaMalloc(&d_output, img_size);
    
    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);
    
    std::cout << h_output[n - 1] << std::endl;
    std::cout << milliseconds << std::endl;
    
    cudaFree(d_image); cudaFree(d_mask); cudaFree(d_output);
    delete[] h_image; delete[] h_output; delete[] h_mask;
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    return 0;
}