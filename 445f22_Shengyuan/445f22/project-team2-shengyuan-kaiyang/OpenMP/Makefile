CC = gcc

all: floyd_openMP

floyd_openMP: APSPtest.o MatUtil.o Floyd_openMP.o getCommandLine.o
	$(CC) -fopenmp -o floyd_openMP APSPtest.o MatUtil.o Floyd_openMP.o getCommandLine.o

%.o: %.c
	$(CC) -fopenmp -std=c99 -c $<

getCommandLine.o: ../utils/getCommandLine.c
	$(CC) -c -o getCommandLine.o ../utils/getCommandLine.c
clean:
	rm *.o floyd_openMP
