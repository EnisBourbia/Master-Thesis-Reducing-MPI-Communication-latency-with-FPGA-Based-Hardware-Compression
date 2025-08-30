#!/bin/bash
mpicc -fPIC -shared -o liblogmpi.so log_mpi.c
LD_PRELOAD=./liblogmpi.so mpirun -n 4 --oversubscribe python3 train_lm.py
