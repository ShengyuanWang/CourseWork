# Makefile for countSort examples
#

CPP = g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4

all: getopt countSort_seq countSort_omp

countSort_seq: countSort_seq.c seq_time.h
	${CPP} -o countSort_seq countSort_seq.c

countSort_omp: countSort_omp.c
	${CPP} ${OMP} -o countSort_omp countSort_omp.c -I$(INCLUDE) -l$(LIB)

getopt: getopt.c
	${CPP} -o getopt getopt.c 

clean:
	rm -f countSort_seq countSort_omp getopt
	