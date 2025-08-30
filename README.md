# Vitis-Accelerated Hardware Compression & MPI Simulation

This repo has two parts:

1) **Hardware-accelerated compression** using the AMD Vitis™ Data Compression Library (L2/L3 demos).  
2) **MPI communication tracing → SimGrid** workflow to simulate end-to-end communication with traces captured from a real MPI run.

#Link to Hardware-accelerated files [Click here to download](https://drive.google.com/file/d/1tE3bS2hZoHN-gbjuv16H0iMI9eLesWgz/view?usp=drive_link)

The hardware build/run pieces **follow AMD/Xilinx’s documented Vitis + XRT flows** (software emulation, hardware emulation, and hardware). The MPI tracing and conversion scripts live alongside and are independent of the compression kernels.

---

## Prerequisites

- **Vitis** (matching your platform) and **XRT** installed on the host.
- **Platform files** (e.g., `xilinx_u280_gen3x16_xdma_1_202211_1`) installed and discoverable.
- **C/C++ toolchain** for host code.
- **OpenMPI/MPICH** (for running MPI jobs).
- **Python 3.8+** with standard library (no external deps required for scripts).

> On a typical install you’ll have:
>
> - Vitis tools at `/opt/Xilinx/Vitis/<version>/`
> - XRT at `/opt/xilinx/xrt/`

---

## Environment setup (Vitis + XRT)

```bash
# 1) Vitis tools (sets $XILINX_VITIS and $XILINX_VIVADO)
source /opt/Xilinx/Vitis/<VERSION>/settings64.sh

# 2) XRT runtime (sets $XILINX_XRT)
source /opt/xilinx/xrt/setup.sh


If you use an environment module system, you might also see:
module load vitis

sw_emu — Software emulation: fastest build/run, functional only.

hw_emu — Hardware emulation: cycle-approximate simulation of your kernel; much slower; recommended to use small datasets.

hw — Hardware bitstream: compile & run on the physical card (time-consuming build).

# For software emulation
export XCL_EMULATION_MODE=sw_emu

# For hardware emulation
export XCL_EMULATION_MODE=hw_emu

# Build & run: Software emulation
cd /L3/demos/<algorithm>

# Common pattern used by library demos:
make run TARGET=sw_emu DEVICE=/path/to/<platform>/

# Or if you build host/xclbin separately:
make host
make xclbin TARGET=sw_emu DEVICE=/path/to/<platform>/

export XCL_EMULATION_MODE=sw_emu
./build/<your_host_exe> <your>.sw_emu.<platform>.xclbin [args...]

# Build & run: Hardware emulation
cd /L3/demos/<algorithm>

# One-time (per platform) emulation config:
emconfigutil --platform <platform_name>
# Ensure emconfig.json is in your run directory (copy if needed):
cp emconfig.json ./build/  # or wherever you run the host from

# Build for hardware emulation:
make run TARGET=hw_emu DEVICE=/path/to/<platform>/
# or:
make host
make xclbin TARGET=hw_emu DEVICE=/path/to/<platform>/

# Run hw emulation with small datasets (slow but detailed)
export XCL_EMULATION_MODE=hw_emu
./build/<your_host_exe> <your>.hw_emu.<platform>.xclbin [args...]



# XRT profiling & traces (xrt.ini)

Create compression/xrt.ini in the same directory you run the host from (or point to it with XRT_INI_PATH):
[Runtime]
runtime_log = console

[Debug]
# Collect OpenCL timeline & summary
opencl_trace = true
opencl_summary = true
data_transfer_trace = fine
timeline_trace = true

[Emulation]
# Choose 'batch' to run headless or 'gui' to open simulator waveforms (hw_emu)
debug_mode = batch

You can inspect runs in Vitis Analyzer:
# After a run, open the run/link summaries:
vitis_analyzer xrt.run_summary
# or
vitis_analyzer <kernel>.xclbin.link_summary


## MPI communication tracing → SimGrid

This workflow instruments your MPI program to produce per-rank JSONL logs of key MPI calls, then converts those logs into SimGrid trace scripts.

# 1) Build the MPI interposer
cd system_simulation_tools

# Build a shared object suitable for LD_PRELOAD
gcc -D_GNU_SOURCE -fPIC -shared -o libmpi_trace.so interpose_mpi.c -ldl

What it does:

Interposes MPI_Send, MPI_Recv, and MPI_Allreduce.

Writes JSONL to mpi_trace_rank_<R>.txt (one file per rank).

Each record contains fields like: event_id, func, timestamp_ns, count, datatype, bytes, src, dest, direction, tag, op, comm_size, rank, and a hex dump of data.

# 2) Run your MPI job with the interposer
# Example: 4 ranks
mpirun -np 4 -x LD_PRELOAD=$PWD/libmpi_trace.so -x LD_BIND_NOW=1 ./your_mpi_program [args...]

Outputs (per rank):
mpi_trace_rank_0.txt
mpi_trace_rank_1.txt
mpi_trace_rank_2.txt
mpi_trace_rank_3.txt

# Optional: split merged JSONL logs (one file with many ranks)
python3 extract_logs_for_rank.py mpi_trace.log 1 logs_rank1/

RANK_COUNT = 4           # number of ranks in your run

python3 convert_to_simgrid.py


This will produce, for each rank:
rank_0.trace
rank_1.trace
rank_2.trace
rank_3.trace

Each .trace contains simple operations:
compute <seconds>
send <bytes> <dest_rank>
recv <bytes> <src_rank>
allreduce <bytes>


