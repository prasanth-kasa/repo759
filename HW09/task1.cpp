#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "cluster.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n t\n";
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);
    const int t = std::atoi(argv[2]);
    if (n_ll <= 0 || t < 1 || t > 10) {
        std::cerr << "Error: n must be > 0 and t must be in [1, 10].\n";
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(n_ll);
    omp_set_num_threads(t);

    std::vector<float> arr(n);
    std::mt19937 rng(759);
    std::uniform_real_distribution<float> dist(0.0f, static_cast<float>(n));
    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(rng);
    }
    std::sort(arr.begin(), arr.end());

    std::vector<float> centers(static_cast<std::size_t>(t));
    const float n_f = static_cast<float>(n);
    const float t_f = static_cast<float>(t);
    for (int i = 0; i < t; ++i) {
        centers[static_cast<std::size_t>(i)] =
            (2.0f * static_cast<float>(i) + 1.0f) * n_f / (2.0f * t_f);
    }

    std::vector<float> dists(static_cast<std::size_t>(t), 0.0f);

    // Warmup: pages, caches, frequency — not timed; reset dists before measured run.
    cluster(n, static_cast<std::size_t>(t), arr.data(), centers.data(), dists.data());
    std::fill(dists.begin(), dists.end(), 0.0f);

    const auto start = std::chrono::high_resolution_clock::now();
    cluster(n, static_cast<std::size_t>(t), arr.data(), centers.data(), dists.data());
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed_ms = end - start;

    std::size_t argmax = 0;
    float max_dist = dists[0];
    for (std::size_t i = 1; i < static_cast<std::size_t>(t); ++i) {
        if (dists[i] > max_dist) {
            max_dist = dists[i];
            argmax = i;
        }
    }

    std::cout << max_dist << '\n';
    std::cout << argmax << '\n';
    std::cout << elapsed_ms.count() << '\n';

    return 0;
}
