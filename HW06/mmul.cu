#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n) {
    // We want C = alpha * A * B + beta * C. 
    // Setting alpha = 1.0 and beta = 1.0 achieves C = AB + C.
    const float alpha = 1.0f;
    const float beta = 1.0f;

    // cublasSgemm performs the matrix-matrix multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
    
    // As requested in the header file comments
    cudaDeviceSynchronize();
}