#include "montecarlo.h"

#include <cmath>


int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    const float r2 = radius * radius;
    int count = 0;

#ifdef MONTECARLO_USE_SIMD
#pragma omp parallel for simd reduction(+ : count)
#else
#pragma omp parallel for reduction(+ : count)
#endif
    for (size_t i = 0; i < n; ++i) {
        const float xi = x[i];
        const float yi = y[i];
        if (xi * xi + yi * yi <= r2) {
            ++count;
        }
    }
    return count;
}
