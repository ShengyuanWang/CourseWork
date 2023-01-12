
# Makefile for game of life multicore pgcc version

#compilation with pgcc and flags
CC_MC= pgcc 
GCC= gcc

# options
FAST = -fast
# options  (-w ignores warnings)
MOPTS = -mp -ta=multicore -Minfo=opt -w

#profiling for multicore  and GPU
NVPROF1 = nvprof -f -o
NVPROF2 = nvprof --cpu-profiling off --openacc-profiling on -i 

NUM_ITERS=200

all: gol_mc

######## multicore fast pgcc version
getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(GCC) -c -o getCommandLine.o ../utils/getCommandLine.c

gol_mc: gol_main.c gol_mc.c getCommandLine.o
	$(CC_MC) $(FAST) $(MOPTS) -o gol_mc getCommandLine.o gol_mc.c gol_main.c

gol_mc_prof: gol_mc
	${NVPROF1} gol_mc.prof ./gol_mc -i ${NUM_ITERS} -t 4
	${NVPROF2} gol_mc.prof


####### clean
clean:
	rm -f *.o gol_mc gmon.out profile.out *.prof
