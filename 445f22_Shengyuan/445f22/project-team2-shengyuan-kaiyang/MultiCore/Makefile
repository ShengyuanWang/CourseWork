CC = gcc

CC_MC= pgcc 

# options
FAST = -fast
# options  (-w ignores warnings)
MOPTS = -mp -ta=multicore -Minfo=opt -w

######## multicore fast pgcc version
all: floyd_mc

floyd_mc: APSPtest.c MatUtil.c Floyd_mc.c getCommandLine.o
	$(CC_MC) $(FAST) $(MOPTS) -o floyd_mc APSPtest.c MatUtil.c Floyd_mc.c getCommandLine.o

%.o: %.c
	$(CC) -c -std=c99 -c $<

getCommandLine.o: ../utils/getCommandLine.c
	$(CC) -c -o getCommandLine.o ../utils/getCommandLine.c
clean:
	rm *.o floyd_mc
