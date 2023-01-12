CC=gcc
OMP=-fopenmp


all: trap-seq trap-omp

ParGetCommandLine.o: ./utils/ParGetCommandLine.c ./utils/ParGetCommandLine.h
	$(CC) -c -o ParGetCommandLine.o ./utils/ParGetCommandLine.c

SeqGetCommandLine.o: ./utils/SeqGetCommandLine.c ./utils/SeqGetCommandLine.h
	$(CC) -c -o SeqGetCommandLine.o ./utils/SeqGetCommandLine.c

trap-seq: trap-seq.c SeqGetCommandLine.o
	${CC} ${OMP} -o trap-seq SeqGetCommandLine.o trap-seq.c -lm

trap-omp: trap-omp.c ParGetCommandLine.o
	${CC} ${OMP} -o trap-omp ParGetCommandLine.o trap-omp.c -lm

clean:
	rm -f trap-omp trap-seq SeqGetCommandLine.o ParGetCommandLine.o
