#include "matmul.cuh"
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    extern __shared__ char shared_mem[];
    T *As = (T*)shared_mem;
    T *Bs = (T*)&shared_mem[block_dim * block_dim * sizeof(T)];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * block_dim + ty;
    int col = bx * block_dim + tx;

    T sum = 0;
    int num_tiles = (n + block_dim - 1) / block_dim;

    for (int m = 0; m < num_tiles; ++m) {
        if (row < n && m * block_dim + tx < n)
            As[ty * block_dim + tx] = A[row * n + m * block_dim + tx];
        else
            As[ty * block_dim + tx] = 0;

        if (m * block_dim + ty < n && col < n)
            Bs[ty * block_dim + tx] = B[(m * block_dim + ty) * n + col];
        else
            Bs[ty * block_dim + tx] = 0;

        __syncthreads();

        for (int k = 0; k < block_dim; ++k) {
            sum += As[ty * block_dim + k] * Bs[k * block_dim + tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

template <typename T>
void matmul_wrapper(const T *A, const T *B, T *C, unsigned int n, unsigned int block_dim) {
    dim3 threads(block_dim, block_dim);
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(T);
    
    matmul_kernel<<<blocks, threads, shared_mem_size>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    matmul_wrapper(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    matmul_wrapper(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    matmul_wrapper(A, B, C, n, block_dim);
}
