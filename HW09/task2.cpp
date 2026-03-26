#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

#include "montecarlo.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n t\n";
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);
    const int t = std::atoi(argv[2]);
    if (n_ll <= 0 || t < 1 || t > 10) {
        std::cerr << "Error: n must be > 0 and t must be in [1, 10].\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(n_ll);
    constexpr float radius = 1.0f;
    omp_set_num_threads(t);

    std::vector<float> x(n);
    std::vector<float> y(n);
    std::mt19937 rng(760);
    std::uniform_real_distribution<float> dist(-radius, radius);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    const int incircle = montecarlo(n, x.data(), y.data(), radius);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    const double pi_est = 4.0 * static_cast<double>(incircle) / static_cast<double>(n);
    std::cout << pi_est << '\n';
    std::cout << elapsed_ms.count() << '\n';

    return 0;
}
