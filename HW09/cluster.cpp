#include "cluster.h"

#include <cmath>
#include <vector>

namespace {
// Enough floats to keep each thread's slot on its own cache line (avoid false sharing).
constexpr int kLineFloats = 64 / static_cast<int>(sizeof(float));
}  // namespace

void cluster(const size_t n, const size_t t, const float *arr, const float *centers,
             float *dists) {
    const int nt = static_cast<int>(t);
    const size_t chunk = n / t;

    std::vector<float> partial(t * static_cast<size_t>(kLineFloats), 0.0f);

#pragma omp parallel num_threads(nt)
    {
        const int tid = omp_get_thread_num();
        const size_t start = static_cast<size_t>(tid) * chunk;
        const size_t end = start + chunk;
        float sum = 0.0f;
        for (size_t i = start; i < end; ++i) {
            sum += std::fabs(arr[i] - centers[tid]);
        }
        partial[static_cast<size_t>(tid) * kLineFloats] = sum;
    }

    for (size_t i = 0; i < t; ++i) {
        dists[i] = partial[i * static_cast<size_t>(kLineFloats)];
    }
}
