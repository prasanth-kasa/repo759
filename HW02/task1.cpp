#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include "scan.h"

// Use standard namespace for convenience, or prefix std::
using namespace std;

int main(int argc, char* argv[]) {
    // 1. Read n from the first command line argument [cite: 19]
    // Usage check (optional but good practice)
    if (argc < 2) {
        cerr << "Usage: ./task1 n" << endl;
        return 1;
    }

    // Convert argument to size_t (n is a positive integer)
    size_t n = static_cast<size_t>(atol(argv[1]));

    // 2. Allocate memory for the input and output arrays
    // "Deallocates memory when necessary" implies manual management [cite: 24]
    float* arr = new float[n];
    float* output = new float[n];

    // 3. Fill array with random floats between -1.0 and 1.0 [cite: 18]
    // Using Mersenne Twister for better random generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < n; ++i) {
        arr[i] = dis(gen);
    }

    // 4. Scan the array using your scan function [cite: 20]
    // Start timing
    auto start = chrono::high_resolution_clock::now();

    scan(arr, output, n);

    // Stop timing
    auto end = chrono::high_resolution_clock::now();

    // 5. Prints out the time taken by your scan function in milliseconds [cite: 21]
    chrono::duration<double, milli> duration_ms = end - start;
    cout << duration_ms.count() << endl;

    // 6. Prints the first element of the output scanned array [cite: 22]
    cout << output[0] << endl;

    // 7. Prints the last element of the output scanned array [cite: 23]
    cout << output[n - 1] << endl;

    // 8. Deallocate memory [cite: 24]
    delete[] arr;
    delete[] output;

    return 0;
}