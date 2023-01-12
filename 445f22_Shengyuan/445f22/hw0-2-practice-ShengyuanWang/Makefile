CC = gcc
FLAGS = -fopenmp
TARGETS = hello

all: $(TARGETS)

getCommandLine.o: ./utils/getCommandLine.c ./utils/getCommandLine.h
	$(CC) -c -o getCommandLine.o ./utils/getCommandLine.c

hello: hello.c getCommandLine.o
	$(CC) -o hello getCommandLine.o hello.c $(FLAGS)

clean:
	rm -f $(TARGETS)
	
