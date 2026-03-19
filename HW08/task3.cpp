#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "msort.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./task3 n t ts\n";
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);
    const int t = std::atoi(argv[2]);
    const long long ts_ll = std::atoll(argv[3]);
    if (n_ll <= 0 || t < 1 || t > 20 || ts_ll <= 0) {
        std::cerr << "Error: n > 0, t in [1, 20], and ts > 0 are required.\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(n_ll);
    const std::size_t ts = static_cast<std::size_t>(ts_ll);
    omp_set_num_threads(t);

    std::vector<int> arr(n);
    std::mt19937 rng(761);
    std::uniform_int_distribution<int> dist(-1000, 1000);
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(rng);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    msort(arr.data(), n, ts);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::cout << arr.front() << '\n';
    std::cout << arr.back() << '\n';
    std::cout << elapsed_ms.count() << '\n';

    return 0;
}
