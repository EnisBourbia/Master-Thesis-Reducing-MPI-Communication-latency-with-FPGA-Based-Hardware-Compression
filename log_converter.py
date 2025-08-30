import os
import json
from collections import defaultdict

INPUT_PREFIX = "mpi_trace_rank_"
OUTPUT_SUFFIX = ".trace"
MIN_COMPUTE_GAP_NS = 10_000_000  # 10 ms to filter noise
RANK_COUNT = 4  # Set to number of ranks you used in mpirun

def ns_to_sec(ns):
    return ns / 1e9

def parse_trace_file(rank):
    filename = f"{INPUT_PREFIX}{rank}.txt"
    with open(filename, "r") as f:
        events = [json.loads(line) for line in f if line.strip()]
    return sorted(events, key=lambda e: e["timestamp_ns"])

def write_simgrid_trace(rank, ops):
    with open(f"rank_{rank}{OUTPUT_SUFFIX}", "w") as f:
        for op in ops:
            if op["type"] == "compute":
                f.write(f"compute {op['duration']:.6f}\n")
            elif op["type"] == "allreduce":
                f.write(f"allreduce {op['bytes']}\n")
            elif op["type"] == "send":
                f.write(f"send {op['bytes']} {op['dest']}\n")
            elif op["type"] == "recv":
                f.write(f"recv {op['bytes']} {op['src']}\n")

def convert_all():
    for rank in range(RANK_COUNT-1, RANK_COUNT):
        events = parse_trace_file(rank)
        ops = []
        prev_ts = None

        for e in events:
            ts = e["timestamp_ns"]
            if prev_ts is not None:
                delta = ts - prev_ts
                if delta > MIN_COMPUTE_GAP_NS:
                    ops.append({"type": "compute", "duration": ns_to_sec(delta)})
            prev_ts = ts

            op_type = e["func"]
            bytes_ = e["bytes"]

            if op_type == "MPI_Allreduce":
                ops.append({"type": "allreduce", "bytes": bytes_})
            elif op_type == "MPI_Send":
                ops.append({"type": "send", "bytes": bytes_, "dest": e["dest"]})
            elif op_type == "MPI_Recv":
                ops.append({"type": "recv", "bytes": bytes_, "src": e["src"]})

        write_simgrid_trace(rank, ops)

if __name__ == "__main__":
    convert_all()

