#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <dlfcn.h>
#include <time.h>
#include <string.h>

static FILE* log_file = NULL;
static long event_id_counter = 0;

// Get timestamp in nanoseconds
long long now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// Convert MPI_Datatype to string for logging
const char* mpi_datatype_to_string(MPI_Datatype dtype) {
    if (dtype == MPI_CHAR) return "MPI_CHAR";
    if (dtype == MPI_INT) return "MPI_INT";
    if (dtype == MPI_FLOAT) return "MPI_FLOAT";
    if (dtype == MPI_DOUBLE) return "MPI_DOUBLE";
    if (dtype == MPI_LONG) return "MPI_LONG";
    if (dtype == MPI_BYTE) return "MPI_BYTE";
    return "MPI_UNKNOWN";
}

// Log binary data to file as hex string
void log_buffer_hex(const void* buf, int bytes) {
    const unsigned char* data = (const unsigned char*)buf;
    for (int i = 0; i < bytes; ++i) {
        fprintf(log_file, "%02x", data[i]);
        if (i < bytes - 1) fprintf(log_file, " ");
    }
}

// Log entry to file
void log_event(const char* func, int count, MPI_Datatype datatype, int src, int dest, const void* buf, const char* direction, int tag, const char* op_name, int world_size, int rank, long event_id) {
    if (!log_file) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        char filename[256];
        snprintf(filename, sizeof(filename), "mpi_trace_rank_%d.txt", rank);
        log_file = fopen(filename, "a");
    }
    
    if (!log_file) return;

    int size = 0;
    MPI_Type_size(datatype, &size);
    long long timestamp = now_ns();
    const char* dtype_name = mpi_datatype_to_string(datatype);

    fprintf(log_file,
        "{\"event_id\":%ld,\"func\":\"%s\",\"timestamp_ns\":%lld,\"count\":%d,\"datatype\":\"%s\",\"bytes\":%d,\"src\":%d,\"dest\":%d,\"direction\":\"%s\",\"tag\":%d,\"op\":\"%s\",\"comm_size\":%d,\"rank\":%d,\"data\":\"",
        event_id, func, timestamp, count, dtype_name, count * size, src, dest, direction, tag, op_name ? op_name : "", world_size, rank);

    if (buf) {
        log_buffer_hex(buf, count * size);
    } else {
        fprintf(log_file, "<null>");
    }

    fprintf(log_file, "\"}\n");
    fflush(log_file);
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
    static int (*real_MPI_Send)(const void*, int, MPI_Datatype, int, int, MPI_Comm) = NULL;
    if (!real_MPI_Send) real_MPI_Send = dlsym(RTLD_NEXT, "MPI_Send");

    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    long eid = event_id_counter++;
    log_event("MPI_Send", count, datatype, rank, dest, buf, "send", tag, "", world_size, rank, eid);

    return real_MPI_Send(buf, count, datatype, dest, tag, comm);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status) {
    static int (*real_MPI_Recv)(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*) = NULL;
    if (!real_MPI_Recv) real_MPI_Recv = dlsym(RTLD_NEXT, "MPI_Recv");

    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    int ret = real_MPI_Recv(buf, count, datatype, source, tag, comm, status);
    long eid = event_id_counter++;
    log_event("MPI_Recv", count, datatype, source, rank, buf, "recv", tag, "", world_size, rank, eid);

    return ret;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm) {
    static int (*real_MPI_Allreduce)(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm) = NULL;
    if (!real_MPI_Allreduce) real_MPI_Allreduce = dlsym(RTLD_NEXT, "MPI_Allreduce");

    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    int size = 0;
    MPI_Type_size(datatype, &size);

    void* temp = NULL;
    const void* log_buf = sendbuf;

    if (sendbuf == MPI_IN_PLACE) {
        temp = malloc(count * size);
        if (temp) {
            memcpy(temp, recvbuf, count * size);
            log_buf = temp;
        }
    }

    long eid = event_id_counter++;
    log_event("MPI_Allreduce", count, datatype, rank, -1, log_buf, "send", 0, "MPI_SUM", world_size, rank, eid);
    if (temp) free(temp);

    int result = real_MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    log_event("MPI_Allreduce", count, datatype, -1, rank, recvbuf, "recv", 0, "MPI_SUM", world_size, rank, eid);

    return result;
}

__attribute__((destructor))
void close_log() {
    if (log_file) fclose(log_file);
}
