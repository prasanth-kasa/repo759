#include "scan.cuh"

// The primary scan kernel 
__global__ void hillis_steele(const float* input, float* output, float* block_sums, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    int pout = 0, pin = 1;
    // Load into shared memory, handle out-of-bounds 
    temp[pout * blockDim.x + tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // Hillis-Steele step
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;
        if (tid >= offset) {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid] + temp[pin * blockDim.x + tid - offset];
        } else {
            temp[pout * blockDim.x + tid] = temp[pin * blockDim.x + tid];
        }
        __syncthreads();
    }

    // Write output [cite: 37]
    if (gid < n) {
        output[gid] = temp[pout * blockDim.x + tid];
    }
    
    // Write total block sum for the multi-block step
    if (tid == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = temp[pout * blockDim.x + tid];
    }
}

// Additional kernel to distribute block sums 
__global__ void add_sums(float* output, const float* block_sums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < n) {
        output[gid] += block_sums[blockIdx.x - 1];
    }
}

void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = 2 * threads_per_block * sizeof(float);

    if (num_blocks == 1) {
        hillis_steele<<<1, threads_per_block, shared_mem_size>>>(input, output, nullptr, n);
        cudaDeviceSynchronize();
        return;
    }

    // Allocate additional memory for intermediate block sums [cite: 42]
    float *block_sums, *scanned_block_sums;
    cudaMallocManaged(&block_sums, num_blocks * sizeof(float));
    cudaMallocManaged(&scanned_block_sums, num_blocks * sizeof(float));

    // 1. Scan individual blocks
    hillis_steele<<<num_blocks, threads_per_block, shared_mem_size>>>(input, output, block_sums, n);
    cudaDeviceSynchronize();

    // 2. Scan the array of block sums
    hillis_steele<<<1, threads_per_block, shared_mem_size>>>(block_sums, scanned_block_sums, nullptr, num_blocks);
    cudaDeviceSynchronize();

    // 3. Add the scanned sums to the outputs of the respective blocks
    add_sums<<<num_blocks, threads_per_block>>>(output, scanned_block_sums, n);
    cudaDeviceSynchronize();

    cudaFree(block_sums);
    cudaFree(scanned_block_sums);
}