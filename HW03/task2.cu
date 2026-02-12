#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// Kernel: computes a*x + y
// x = threadIdx.x, y = blockIdx.x
__global__ void affineKernel(int* dA, int a) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    
    // Calculate unique global index to determine where to write
    // Block size is 8, so index = blockIdx.x * 8 + threadIdx.x
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    dA[idx] = a * x + y;
}

int main() {
    const int num_blocks = 2;
    const int threads_per_block = 8;
    const int N = num_blocks * threads_per_block; // 16 integers
    const int bytes = N * sizeof(int);

    // 1. Allocate device memory [cite: 33]
    int* dA;
    cudaMalloc(&dA, bytes);

    // 2. Generate random 'a' [cite: 42, 74]
    // Note: Prompt text contained a typo "100 and 100". 
    // Based on Example Output (0, 10, 20...), 'a' was 10.
    // We will generate a random number in a range (e.g., 10-100) to satisfy requirements.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(10, 100); 
    int a = dist(gen);

    // 3. Launch kernel: 2 blocks, 8 threads [cite: 34]
    affineKernel<<<num_blocks, threads_per_block>>>(dA, a);
    cudaDeviceSynchronize();

    // 4. Copy back to host [cite: 44]
    std::vector<int> hA(N);
    cudaMemcpy(hA.data(), dA, bytes, cudaMemcpyDeviceToHost);

    // 5. Print values space separated [cite: 45]
    // Note: Output depends on the random 'a'.
    std::cout << "Values for a = " << a << ":" << std::endl; // Optional debug print
    for (int i = 0; i < N; i++) {
        std::cout << hA[i] << (i == N - 1 ? "" : " ");
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(dA);
    return 0;
}