// replay_rankpick.cpp  –  each rank reads argv[rank+1]
#include "xbt/replay.hpp"
#include "simgrid/s4u/Actor.hpp"
#include "smpi/smpi.h"
#include <string>

int main(int argc, char* argv[])
{
  /* every actor gets its rank as a property set by smpirun */
  int rank = std::stoi(simgrid::s4u::Actor::self()->get_property("rank"));

  if (argc <= rank + 1) {
    fprintf(stderr,
            "Rank %d did not receive its trace file (argc=%d).\n", rank, argc);
    return 1;
  }
  const char* tracefile = argv[rank + 1];   // <- each rank picks its own

  smpi_replay_init("job1", rank, 0.0);      // common instance_id
  smpi_replay_main(rank, tracefile);        // run
  return 0;                                 // MPI finalised inside
}

