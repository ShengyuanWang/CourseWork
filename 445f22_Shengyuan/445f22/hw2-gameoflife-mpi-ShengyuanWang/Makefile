# DESCRIPTION:  Makefile for serial/parallel codes in C
# AUTHOR:       Aaron Weeden, The Shodor Education Foundation, Inc.
# DATE:         September, 2011

#Updated by Libby Shoop, Macalester College, for COMP 445

# Code prefix
PROGRAM_PREFIX=life

# Compilers and flags
CC=gcc
MPICC=mpicc

#CFLAGS+=-DSHOW_RESULTS # Uncomment to make the program print its results

# Source files
SEQSRC=life_seq.c
MPISRC=life_mpi.c

# MPI
MPI_FLAGS=-DMPI

EXECUTABLES=$(PROGRAM_PREFIX).{serial,mpi}

# Make rules
$(PROGRAM_PREFIX).serial: $(SEQSRC)
	$(CC) $(CFLAGS) $(SEQSRC) -o $(PROGRAM_PREFIX).serial

$(PROGRAM_PREFIX).mpi: $(MPISRC)
	$(MPICC) $(CFLAGS) $(MPI_FLAGS) $(MPISRC) -o $(PROGRAM_PREFIX).mpi


clean:
	rm -f $(EXECUTABLES) *.o

all: life.serial life.mpi

