#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <chrono>
#include "convolution.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: ./task2 n m" << endl;
        return 1;
    }
    
    std::size_t n = static_cast<std::size_t>(atoi(argv[1]));
    std::size_t m = static_cast<std::size_t>(atoi(argv[2]));

    float *image = new float[n * n];
    float *mask = new float[m * m];
    float *output = new float[n * n];

    random_device rd;
    mt19937 gen(rd());
    
    uniform_real_distribution<float> dist_img(-10.0f, 10.0f);
    
    uniform_real_distribution<float> dist_mask(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = dist_img(gen);
    }

    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = dist_mask(gen);
    }

    auto start = high_resolution_clock::now();
    
    convolve(image, output, n, mask, m);
    
    auto end = high_resolution_clock::now();

    duration<double, std::milli> ms_double = end - start;
    
    cout << ms_double.count() << endl;
    
    cout << output[0] << endl;
    
    cout << output[n * n - 1] << endl;

    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}