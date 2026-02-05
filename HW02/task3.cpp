#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "matmul.h"

using namespace std;
using namespace std::chrono;

int main() {

    random_device rd;
    mt19937 gen(rd());
    
    uniform_int_distribution<unsigned int> dist_n(1000, 2000);
    unsigned int n = dist_n(gen);

    uniform_real_distribution<double> dist_val(0.0, 1.0);

    cout << n << endl;

    vector<double> A(n * n);
    vector<double> B(n * n);
    vector<double> C(n * n);

    for (size_t i = 0; i < n * n; ++i) {
        A[i] = dist_val(gen);
        B[i] = dist_val(gen);
    }


    auto start = high_resolution_clock::now();
    mmul1(A.data(), B.data(), C.data(), n);
    auto end = high_resolution_clock::now();
    duration<double, milli> ms = end - start;
    
    cout << ms.count() << endl;
    cout << C.back() << endl;

    start = high_resolution_clock::now();
    mmul2(A.data(), B.data(), C.data(), n);
    end = high_resolution_clock::now();
    ms = end - start;
    
    cout << ms.count() << endl;
    cout << C.back() << endl;

    start = high_resolution_clock::now();
    mmul3(A.data(), B.data(), C.data(), n);
    end = high_resolution_clock::now();
    ms = end - start;
    
    cout << ms.count() << endl;
    cout << C.back() << endl;

    start = high_resolution_clock::now();
    mmul4(A, B, C.data(), n);
    end = high_resolution_clock::now();
    ms = end - start;
    
    cout << ms.count() << endl;
    cout << C.back() << endl;

    return 0;
}