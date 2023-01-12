PROG = calcPiSeq
CC = g++
CFLAGS = -Wall -ansi -pedantic -std=c++11
LFLAGS = -o $(PROG)
INCLUDE=/usr/local/include/trng
LIB=trng4

$(PROG): $(PROG).C
	$(CC) $(CFLAGS) $(PROG).C $(LFLAGS) -I$(INCLUDE) -l$(LIB)

clean:
	rm -f $(PROG) *.o *~ *#
