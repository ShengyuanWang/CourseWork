
# Makefile for game of life multicore pgcc version

#compilation with gcc
GCC= gcc -fopenmp

all: gol_omp

######## multicore fast pgcc version
getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

gol_omp: gol_main.c gol_omp.c getCommandLine.o
	$(GCC)  getCommandLine.o gol_omp.c gol_main.c -o gol_omp


####### clean
clean:
	rm -f *.o gol_omp
