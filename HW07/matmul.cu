#include "matmul.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>

__global__ void mmul_cuda_kernel(const half* A, const half* B, half* C, size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n * n;
    if (idx >= total) {
        return;
    }

    const size_t row = idx / n;
    const size_t col = idx % n;

    float sum = 0.0f;
    for (size_t k = 0; k < n; ++k) {
        sum += __half2float(A[row * n + k]) * __half2float(B[k * n + col]);
    }
    C[idx] = __float2half(sum);
}

void mmul_cuda(const half* A, const half* B, half* C, size_t n, unsigned int threads_per_block) {
    const size_t total = n * n;
    const size_t blocks = (total + threads_per_block - 1) / threads_per_block;
    mmul_cuda_kernel<<<blocks, threads_per_block>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__global__ void mmul_wmma_kernel(const half* A, const half* B, half* C, size_t n) {
    using namespace nvcuda;

    const int tile_row = static_cast<int>(blockIdx.y) * 16;
    const int tile_col = static_cast<int>(blockIdx.x) * 16;
    if (tile_row >= static_cast<int>(n) || tile_col >= static_cast<int>(n)) {
        return;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (size_t k = 0; k < n; k += 16) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(tile_row) * n + k, n);
        wmma::load_matrix_sync(b_frag, B + k * n + static_cast<size_t>(tile_col), n);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __shared__ float tile_buffer[16 * 16];
    wmma::store_matrix_sync(tile_buffer, c_frag, 16, wmma::mem_row_major);

    for (int i = static_cast<int>(threadIdx.x); i < 16 * 16; i += blockDim.x) {
        const int local_row = i / 16;
        const int local_col = i % 16;
        C[(static_cast<size_t>(tile_row) + local_row) * n + static_cast<size_t>(tile_col) + local_col] =
            __float2half(tile_buffer[i]);
    }
}

void mmul_wmma(const half* A, const half* B, half* C, size_t n, unsigned int threads_per_block) {
    (void)threads_per_block;

    // WMMA 16x16x16 tiles require dimensions divisible by 16.
    if (n % 16 != 0) {
        mmul_cuda(A, B, C, n, WARP_SIZE);
        return;
    }

    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    mmul_wmma_kernel<<<blocks, WARP_SIZE>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

void mmul_cublas(const half* A, const half* B, half* C, size_t n) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major by default; swapping A/B produces row-major C = A * B.
    cublasGemmEx(handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 static_cast<int>(n),
                 static_cast<int>(n),
                 static_cast<int>(n),
                 &alpha,
                 B,
                 CUDA_R_16F,
                 static_cast<int>(n),
                 A,
                 CUDA_R_16F,
                 static_cast<int>(n),
                 &beta,
                 C,
                 CUDA_R_16F,
                 static_cast<int>(n),
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaDeviceSynchronize();
    cublasDestroy(handle);
}
