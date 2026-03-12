#include <omp.h>

#include <iostream>

static unsigned long long factorial(int x) {
    unsigned long long result = 1;
    for (int i = 2; i <= x; ++i) {
        result *= static_cast<unsigned long long>(i);
    }
    return result;
}

int main() {
#pragma omp parallel num_threads(4)
    {
#pragma omp single
        {
            std::cout << "Number of threads: " << omp_get_num_threads() << '\n';
        }
        std::cout << "I am thread No. " << omp_get_thread_num() << '\n';
    }

#pragma omp parallel for num_threads(4) schedule(static)
    for (int x = 1; x <= 8; ++x) {
        const unsigned long long value = factorial(x);
#pragma omp critical
        {
            std::cout << x << "!=" << value << '\n';
        }
    }

    return 0;
}
