# Makefile for countSort examples
#

CC=gcc 
CPP=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4
TARGETS = countSort_seq countSort_omp

all: $(TARGETS)

getCommandLine.o: ./utils/getCommandLine.c ./utils/getCommandLine.h
	$(CPP) -c -o getCommandLine.o ./utils/getCommandLine.c

countSort_seq: countSort_seq.c getCommandLine.o
	${CPP} -o countSort_seq getCommandLine.o countSort_seq.c ${OMP}

countSort_omp: countSort_omp.c
	${CPP}  -o countSort_omp getCommandLine.o countSort_omp.c ${OMP} -I$(INCLUDE) -l$(LIB)

clean:
	rm -f $(TARGETS) getCommandLine.o
	