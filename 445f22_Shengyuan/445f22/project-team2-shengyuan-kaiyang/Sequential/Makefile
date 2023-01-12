CC = gcc

all: floyd_seq

floyd_seq: floyd_seq.o MatUtil.o getCommandLine.o
	$(CC) -fopenmp -o floyd_seq floyd_seq.o MatUtil.o getCommandLine.o

%.o: %.c
	$(CC) -fopenmp -std=c99 -c $<

getCommandLine.o: ../utils/getCommandLine.c
	$(CC) -c -o getCommandLine.o ../utils/getCommandLine.c
clean:
	rm *.o floyd_seq
