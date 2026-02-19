#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {    
    extern __shared__ float s[];
    
    float* s_mask = s;
    float* s_image = (float*)&s_mask[2 * R + 1];
    float* s_out = (float*)&s_image[blockDim.x + 2 * R];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 2 * R + 1) {
        s_mask[tid] = mask[tid];
    }
    
    if (gid < n) {
        s_image[R + tid] = image[gid];
    } else {
        s_image[R + tid] = 1.0f;
    }
    
    if (tid < R) {
        int left_idx = gid - (int)R;
        s_image[tid] = (left_idx >= 0) ? image[left_idx] : 1.0f;
    }
    
    if (tid >= blockDim.x - R) {
        int right_idx = gid + (int)R;
        s_image[R + tid + R] = (right_idx < n) ? image[right_idx] : 1.0f;
    }
    
    __syncthreads();
    
    if (gid < n) {
        float sum = 0.0f;
        for (int j = -(int)R; j <= (int)R; ++j) {
            sum += s_image[R + tid + j] * s_mask[j + R]; [cite: 41, 42]
        }
        s_out[tid] = sum;
    }
    
    __syncthreads();
    
    if (gid < n) {
        output[gid] = s_out[tid];
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    size_t shmem_size = ((2 * R + 1) + (threads_per_block + 2 * R) + threads_per_block) * sizeof(float);
    
    stencil_kernel<<<blocks, threads_per_block, shmem_size>>>(image, mask, output, n, R);
}