#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "Error: run with exactly 2 MPI processes (e.g. srun -n 2 ./task3 n).\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (argc != 2) {
        if (rank == 0) {
            std::cerr << "Usage: srun -n 2 ./task3 n\n";
        }
        MPI_Finalize();
        return 1;
    }

    const long long n_ll = std::atoll(argv[1]);
    if (n_ll <= 0) {
        if (rank == 0) {
            std::cerr << "Error: n must be > 0.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int n = static_cast<int>(n_ll);
    std::vector<float> sendbuf(static_cast<std::size_t>(n));
    std::vector<float> recvbuf(static_cast<std::size_t>(n));

    std::mt19937 rng(761u + static_cast<unsigned>(rank));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < sendbuf.size(); ++i) {
        sendbuf[i] = dist(rng);
    }

    constexpr int data_tag = 0;
    constexpr int time_tag = 1;
    MPI_Status status{};

    double t_rank_ms = 0.0;

    if (rank == 0) {
        const double t0 = MPI_Wtime();
        MPI_Send(sendbuf.data(), n, MPI_FLOAT, 1, data_tag, MPI_COMM_WORLD);
        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, 1, data_tag, MPI_COMM_WORLD, &status);
        const double t1 = MPI_Wtime();
        t_rank_ms = (t1 - t0) * 1000.0;

        double other_ms = 0.0;
        MPI_Recv(&other_ms, 1, MPI_DOUBLE, 1, time_tag, MPI_COMM_WORLD, &status);
        const double total_ms = t_rank_ms + other_ms;
        std::cout << total_ms << '\n';
    } else {
        const double t0 = MPI_Wtime();
        MPI_Recv(recvbuf.data(), n, MPI_FLOAT, 0, data_tag, MPI_COMM_WORLD, &status);
        MPI_Send(sendbuf.data(), n, MPI_FLOAT, 0, data_tag, MPI_COMM_WORLD);
        const double t1 = MPI_Wtime();
        t_rank_ms = (t1 - t0) * 1000.0;

        MPI_Send(&t_rank_ms, 1, MPI_DOUBLE, 0, time_tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
