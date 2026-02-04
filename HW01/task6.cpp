#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    for (int i = 0; i <= N; ++i) {
        printf("%d%c", i, (i == N) ? '\n' : ' ');
    }

    for (int i = N; i >= 0; --i) {
        std::cout << i << (i == 0 ? "\n" : " ");
    }

    return 0;
}