#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task2 n t\n";
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);
    const int t = std::atoi(argv[2]);
    if (n_ll <= 0 || t < 1 || t > 20) {
        std::cerr << "Error: n must be > 0 and t must be in [1, 20].\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(n_ll);
    constexpr std::size_t m = 3;
    omp_set_num_threads(t);

    std::vector<float> image(n * n);
    std::vector<float> output(n * n, 0.0f);
    std::vector<float> mask(m * m);

    std::mt19937 rng(760);
    std::uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    std::uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = dist_img(rng);
    }
    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = dist_mask(rng);
    }

    const auto start = std::chrono::high_resolution_clock::now();
    convolve(image.data(), output.data(), n, mask.data(), m);
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::cout << output.front() << '\n';
    std::cout << output.back() << '\n';
    std::cout << elapsed_ms.count() << '\n';

    return 0;
}
