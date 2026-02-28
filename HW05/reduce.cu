#include "reduce.cuh"
#include <cuda_runtime.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float mySum = (i < n) ? g_idata[i] : 0.0f;
    if (i + blockDim.x < n) {
        mySum += g_idata[i + blockDim.x];
    }
    sdata[tid] = mySum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    unsigned int current_N = N;
    float *d_in = *input;
    float *d_out = *output;

    while (current_N > 1) {
        unsigned int blocks = (current_N + (threads_per_block * 2) - 1) / (threads_per_block * 2);
        size_t smem_size = threads_per_block * sizeof(float);
        
        reduce_kernel<<<blocks, threads_per_block, smem_size>>>(d_in, d_out, current_N);
        cudaDeviceSynchronize();
        
        current_N = blocks;
        
        float *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }
    
    if (d_in != *input) {
        cudaMemcpy(*input, d_in, sizeof(float), cudaMemcpyDeviceToDevice);
    }
}