#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    long n_signed = static_cast<long>(n);
    long m_signed = static_cast<long>(m);
    long offset = (m_signed - 1) / 2;

    for (long x = 0; x < n_signed; ++x) {
        for (long y = 0; y < n_signed; ++y) {
            
            float sum = 0.0f;

            for (long i = 0; i < m_signed; ++i) {
                for (long j = 0; j < m_signed; ++j) {
                    
                    long x_im = x + i - offset;
                    long y_im = y + j - offset;

                    float val = 0.0f;

                    bool x_valid = (x_im >= 0 && x_im < n_signed);
                    bool y_valid = (y_im >= 0 && y_im < n_signed);

                    if (x_valid && y_valid) {
                        val = image[x_im * n + y_im];
                    } else if (!x_valid && !y_valid) {
                        val = 0.0f;
                    } else {
                        val = 1.0f;
                    }

                    sum += mask[i * m + j] * val;
                }
            }

            output[x * n + y] = sum;
        }
    }
}