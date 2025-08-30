#!/usr/bin/env python3

import json
import sys

def analyze_file(filename):
    total_bytes = 0
    zero_bytes = 0

    with open(filename, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                data_field = entry.get("data", "")
                if data_field == "<null>":
                    continue
                # Parse hex string into bytes
                bytes_list = [int(b, 16) for b in data_field.split()]
                entry_total = len(bytes_list)
                entry_zero = sum(1 for b in bytes_list if b == 0)
                if entry_total > 0:
                    entry_zero_ratio = entry_zero / entry_total * 100
                    print(f"Entry {line_number}: {entry_zero}/{entry_total} bytes are zero ({entry_zero_ratio:.2f}%)")
                total_bytes += entry_total
                zero_bytes += entry_zero
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_number}.")

    if total_bytes == 0:
        print(f"{filename}: No data to analyze.")
        return

    zero_ratio = zero_bytes / total_bytes * 100
    print(f"\n{filename}: {zero_bytes}/{total_bytes} bytes are zero ({zero_ratio:.2f}%)")

if __name__ == "__main__":
   analyze_file('mpi_trace_rank_3.txt')