
# Makefile for game of life multicore pgcc version

GCC=gcc
#compilation with pgcc and flags
PGCC= pgcc 

# options
FAST = -fast
# options  (-w ignores warnings)
MOPTS = -acc -ta=tesla:cc75,managed -Minfo=accel -w

#profiling for multicore  and GPU
# NVPROF1 = nvprof -f -o
# NVPROF2 = nvprof --cpu-profiling off --openacc-profiling on -i 

all: gol_acc

######## multicore fast pgcc version
getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

gol_acc: gol_main.c gol_acc.c getCommandLine.o
	$(PGCC) $(FAST) $(MOPTS) -o gol_acc getCommandLine.o gol_acc.c gol_main.c


####### clean
clean:
	rm -f *.o gol_acc 
