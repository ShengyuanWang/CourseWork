
# Makefile for game of life

# compilation with gcc for sequential version profiling
GCC_SEQ_PROF= gcc -pg -fopenmp

# gcc
CC_SEQ= gcc -fopenmp

#compilation with pgcc and flags
# CC_SEQ= pgcc -fast -Minfo=opt

all: gol_seq_prof gol_seq

##########################  sequential profile versions
# separate file to be used with all versions of this code
getCommandLine_pg.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(GCC_SEQ_PROF) -c -o getCommandLine_pg.o ../utils/getCommandLine.c

gol_seq_pg.o: gol_seq.c
	$(GCC_SEQ_PROF) -c -o gol_seq_pg.o gol_seq.c

#### sequential gcc for profiling
gol_seq_prof: gol_main.c gol_seq_pg.o getCommandLine_pg.o
	$(GCC_SEQ_PROF) -o gol_seq_prof getCommandLine_pg.o  gol_seq_pg.o gol_main.c

##########################  sequential fast pgcc version
# separate file to be used with all versions of this code
getCommandLine.o: ../utils/getCommandLine.c ../utils/getCommandLine.h
	$(CC_SEQ) -c -o getCommandLine.o ../utils/getCommandLine.c

gol_seq.o: gol_seq.c
	$(CC_SEQ) -c -o gol_seq.o gol_seq.c
 
gol_seq: gol_main.c gol_seq.o getCommandLine.o
	$(CC_SEQ) -o gol_seq getCommandLine.o  gol_seq.o gol_main.c

####### clean
clean:
	rm -f *.o gol_seq_prof gol_seq gmon.out profile.out *.prof
