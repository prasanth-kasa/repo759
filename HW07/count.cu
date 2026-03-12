#include "count.cuh"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

void count(const thrust::device_vector<int>& d_in,
           thrust::device_vector<int>& values,
           thrust::device_vector<int>& counts) {
    if (d_in.empty()) {
        values.clear();
        counts.clear();
        return;
    }

    thrust::device_vector<int> sorted = d_in;
    thrust::sort(sorted.begin(), sorted.end());

    values.resize(sorted.size());
    counts.resize(sorted.size());

    auto end_pair = thrust::reduce_by_key(
        sorted.begin(),
        sorted.end(),
        thrust::constant_iterator<int>(1),
        values.begin(),
        counts.begin());

    const size_t unique_count = static_cast<size_t>(end_pair.first - values.begin());
    values.resize(unique_count);
    counts.resize(unique_count);
}
