#include <stdio.h>
#include <stdlib.h>
#include <omp.h>   // to use openmp timing functions

#include "../utils/getCommandLine.h"

// functions from another file that is compiled in.
// see Makefile for how each version is built.
void init_ghosts(int *grid, int *newGrid, int dim);
int apply_rules(int *grid, int *newGrid, int dim);
void update_grid(int *grid, int *newGrid, int dim);
int gol(int *grid, int *newGrid, int dim);
void init_grid(int *grid, int dim);

//////////////////////////////////////////////////////////////////////
////               main program
//////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    int i, j;
    
    // default values
    int dim = 2048;      //num rows and cols (square grid used)
    int numThreads = 1;  // never used in the sequential version
    // number of game steps
    int itEnd = 1 << 11; // use smaller when testing sequential version
    int verbose = 0;     // not used yet

    // command line
    getArguments(argc, argv, &dim, &itEnd, &verbose, &numThreads);
    if (numThreads != 1) {
      numThreads = 1;     // not used below
      printf("Warning: this is a GPU version and the -t option isn't used.\n");
    }
        
    // grid array with dimension dim + ghost columns and rows
    int    arraySize = (dim+2) * (dim+2);
    size_t bytes     = arraySize * sizeof(int);
    int    *grid     = (int*)malloc(bytes);
 
    // allocate result grid
    int */*restrict*/newGrid = (int*) malloc(bytes);
    init_grid(grid, dim);

    int total = 0; // total number of alive cells    
    int it;

    double st = omp_get_wtime(); // start timing

    // The main loop of iterations to change the grid from
    // one 'time step' to the next.
    int sub_total = 0;
    for(it = 0; it < itEnd; it++){
        sub_total = gol( grid, newGrid, dim );  // one iteration
        // gol( grid, newGrid, dim );

        if (verbose) {
          if (it % 50 == 0) printf("%6d, %d\n", it, sub_total);
        }
    }
    
    // sum up alive cells
    // only done one time at the end on the host, so leave
    // as sequential.
    for (i = 1; i <= dim; i++) {
        for (j = 1; j <= dim; j++) {
            total += grid[i*(dim+2) + j];
        }
    }
 
    printf("Total Alive: %d\n", total);

    // report overall time
    double runtime = omp_get_wtime() - st;
    printf(" total time: %f s\n", runtime);

    free(grid);
    free(newGrid);
 
    return 0;
}
