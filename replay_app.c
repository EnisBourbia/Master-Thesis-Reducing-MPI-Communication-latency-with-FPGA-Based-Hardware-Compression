#include "smpi/smpi.h"
#include <mpi.h>
#include <stdio.h>

XBT_MAIN;  // ⬅ REQUIRED for SimGrid to locate your actor function

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char filename[64];
    snprintf(filename, sizeof(filename), "rank_%d.trace", rank);

    smpi_replay_run("default", rank, 0.0, filename);

    MPI_Finalize();
    return 0;
}

