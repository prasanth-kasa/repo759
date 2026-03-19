#include "msort.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

void merge_ranges(int* arr, int* tmp, std::size_t left, std::size_t mid, std::size_t right) {
    std::size_t i = left;
    std::size_t j = mid;
    std::size_t k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            tmp[k++] = arr[i++];
        } else {
            tmp[k++] = arr[j++];
        }
    }
    while (i < mid) {
        tmp[k++] = arr[i++];
    }
    while (j < right) {
        tmp[k++] = arr[j++];
    }
    for (std::size_t p = left; p < right; ++p) {
        arr[p] = tmp[p];
    }
}

void msort_impl(int* arr, int* tmp, std::size_t left, std::size_t right, std::size_t threshold) {
    const std::size_t len = right - left;
    if (len <= 1) {
        return;
    }
    if (len <= threshold) {
        std::sort(arr + left, arr + right);
        return;
    }

    const std::size_t mid = left + (len / 2);

    #pragma omp task shared(arr, tmp) if(len > threshold)
    msort_impl(arr, tmp, left, mid, threshold);

    #pragma omp task shared(arr, tmp) if(len > threshold)
    msort_impl(arr, tmp, mid, right, threshold);

    #pragma omp taskwait
    merge_ranges(arr, tmp, left, mid, right);
}

}  // namespace

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    if (n <= 1) {
        return;
    }

    const std::size_t threshold_safe = std::max<std::size_t>(2, threshold);
    std::vector<int> tmp(n);

    #pragma omp parallel
    {
        #pragma omp single nowait
        msort_impl(arr, tmp.data(), 0, n, threshold_safe);
    }
}
