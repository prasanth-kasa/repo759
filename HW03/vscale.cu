#include "vscale.cuh"
#include <cuda_runtime.h>

__global__ void vscale(const float *a, float *b, unsigned int n) {
    
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        b[idx] = a[idx] * b[idx];
    }
}