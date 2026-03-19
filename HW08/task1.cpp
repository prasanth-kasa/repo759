#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "matmul.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n t\n";
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);
    const int t = std::atoi(argv[2]);
    if (n_ll <= 0 || t < 1 || t > 20) {
        std::cerr << "Error: n must be > 0 and t must be in [1, 20].\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(n_ll);
    omp_set_num_threads(t);

    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C(n * n, 0.0f);

    std::mt19937 rng(759);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    mmul(A.data(), B.data(), C.data(), n);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::cout << C.front() << '\n';
    std::cout << C.back() << '\n';
    std::cout << elapsed_ms.count() << '\n';

    return 0;
}
