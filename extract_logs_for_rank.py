import json
import os
import sys

def extract_logs_for_rank(log_file_path, rank_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    send_log_path = os.path.join(output_dir, f"rank{rank_id}_send.txt")
    recv_log_path = os.path.join(output_dir, f"rank{rank_id}_recv.txt")

    with open(log_file_path, "r") as log_file, \
         open(send_log_path, "w") as send_out, \
         open(recv_log_path, "w") as recv_out:

        for line in log_file:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip broken lines

            if entry.get("rank") != rank_id:
                continue

            direction = entry.get("direction")
            if direction == "send":
                send_out.write(json.dumps(entry) + "\n")
            elif direction == "recv":
                recv_out.write(json.dumps(entry) + "\n")

    print(f"✅ Logs for rank {rank_id} saved to:\n  📤 {send_log_path}\n  📥 {recv_log_path}")

# Example usage:
# extract_logs_for_rank("mpi_trace.log", 1, "logs_rank1")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python extract_logs_for_rank.py <log_file> <rank_id> <output_dir>")
        sys.exit(1)

    log_file = sys.argv[1]
    rank_id = int(sys.argv[2])
    output_dir = sys.argv[3]

    extract_logs_for_rank(log_file, rank_id, output_dir)

# EXAMPLE RUN: $ python3 extract_logs_for_rank.py mpi_trace.log 1 ./logs_rank1
