CXX=g++
OMP=-fopenmp
INCLUDE=/usr/local/include/trng
LIB=trng4

OBJS=  getCommandLine.o ising_utils.o plotFile.o 2d_matrix_init.o

all: 2d_matrix_init

%.o: %.cpp 
	$(CXX) ${OMP} -c -o $@ $< -I$(INCLUDE) -l$(LIB)

2d_matrix_init: $(OBJS)
	$(CXX) ${OMP} -o 2d_matrix_init $(OBJS) -I$(INCLUDE) -l$(LIB)


######### clean
clean:
		rm -f $(OBJS) 2d_matrix_init
