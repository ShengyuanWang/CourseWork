CC = gcc

#compilation with pgcc and flags
PGCC= pgcc 

# options
FAST = -fast
# options  (-w ignores warnings)
MOPTS = -acc -ta=tesla:cc75,managed -Minfo=accel -w

#profiling for multicore  and GPU
# NVPROF1 = nvprof -f -o
# NVPROF2 = nvprof --cpu-profiling off --openacc-profiling on -i 

all: floyd_acc

floyd_acc: APSPtest.c MatUtil.c Floyd_acc.c getCommandLine.o
	$(PGCC) $(FAST) $(MOPTS) -o floyd_acc APSPtest.c MatUtil.c Floyd_acc.c getCommandLine.o

%.o: %.c
	$(CC) -o -std=c99 -c $<

getCommandLine.o: ../utils/getCommandLine.c
	$(CC) -c -o getCommandLine.o ../utils/getCommandLine.c
clean:
	rm *.o floyd_acc
