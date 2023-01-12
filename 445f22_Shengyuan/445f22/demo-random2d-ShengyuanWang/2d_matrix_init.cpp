//
// This code is loosely based on the original by John Burkardt found here:
// https://people.sc.fsu.edu/~jburkardt/c_src/ising_2d_simulation/ising_2d_simulation.html
//
// Last Modified by: Libby Shoop, Macalester College
// On:  Sept. 27, 2022
//

# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>

#include <omp.h>

#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

#include "getCommandLine.hpp"
#include "ising_utils.hpp"
#include "plotFile.hpp"

// functions found in the code below
int main ( int argc, char *argv[] );

int *ising_2d_initialize ( int m, int n, double thresh, long unsigned int seed, int verbose, int numThreads);
void dbl_mat_uniform_01 ( int m, int n, long unsigned int seed, double r[], int verbose );
void parallel_dbl_mat_uniform_01( int m, int n, long unsigned int seed, double r[], int verbose, int numThreads );


/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for generating 2D matrix for eventual ISING_2D_SIMULATION.

  Usage:

    ising_2d_simulation  m  n  iterations  thresh  seed

    * M, N, the number of rows and columns.
    * ITERATIONS, the number of iterations.
    * THRESH, the threshhold.
    * SEED, a seed for the random number generator.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 June 2013

  Author:

    John Burkardt
*/
{
  int *c1;
  int i;
  int iterations;    // number of iterations of simulation
  int m;     // number of columns in the matrix
  int n;     // number of rows in the matrix
  // table of probabilities
  double prob[5] = { 0.98, 0.85, 0.50, 0.15, 0.02 };

  // the threshold over which the cell is initialized to 1
  // and under or equal to which the cell is initialzed to -1
  double thresh;

  int useConstantSeed;  // constant seed for testing

  // verbose flag for printing information to debug whether the result
  // from parallelizing is correct.
  int verbose;

  int numThreads;  // number of threads for parallelization

  // set default values
  m = 8;
  n = 8;
  iterations = 15;
  thresh = 0.50;
  useConstantSeed = 0;
  verbose = 0;
  numThreads = 1;
   
  // get arguments from command line
  getArguments( argc, argv, &m, &n, &iterations, &thresh, 
                &useConstantSeed, &verbose, &numThreads);

  long unsigned int seed1;
  // long unsigned int seed2; // designed for later use; not used here

  if (useConstantSeed) {
    // fixed seed only for testing/debugging
    seed1 = 1403297347956120;
  } else {
    seed1 = (long unsigned int)time(NULL);
  }

  if (verbose) {
    printSimInputs(m, n, iterations, thresh, seed1, prob);
  }
  
/*
  Initialize the system.
*/
  c1 = ising_2d_initialize ( m, n, thresh, seed1, verbose, numThreads );

/*
  Write the initial state to a gnuplot file.
*/
  const char *init_plot_filename = "ising_2d_initial.txt";
  const char *init_png_filename = "ising_2d_initial.png";
  plot_file ( m, n, c1, "Initial Configuration", 
              init_plot_filename,  init_png_filename);

  if (verbose) {
    printf ( "\n" );
    printf ( "  Created the gnuplot graphics file \"%s\"\n", 
             init_plot_filename );
    printf ("Run gnuplot on this file to create \"%s\"\n", init_png_filename);
  }


  free ( c1 );

  // more code would be here for a full simulation

  return 0;
}
/************************* end of main ***************************************/



/******************************************************************************/
/*
  Purpose:
    ISING_2D_INITIALIZE initializes the Ising array.

  Licensing:
    This code is distributed under the GNU LGPL license. 

  Modified: 23 November 2011

  Author: John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input, double THRESH, the threshhold.

    Input/output, int *SEED, a seed for the random 
    number generator.

    Output, in ISING_2D_INITIALIZE[M*N], the initial Ising array.
*/
int *ising_2d_initialize ( int m, int n, double thresh, long unsigned int seed, int verbose, int numThreads )
{
  int *c1;
  int i;
  int j;
  double *r;

  r = ( double * ) malloc ( m * n * sizeof ( double ) );

// Use function to generate a 'falttened matrix' of random values between
// 0.0 and 1.0.
// added by Libby Shoop
  // sequential version using trng
  dbl_mat_uniform_01( m, n, seed, r, verbose );
  
  // parallel version using trng and leapfrogging
  // TODO: try the parallel version with 1, 2, 4 threads and -c -v
  parallel_dbl_mat_uniform_01( m, n, seed, r, verbose, numThreads );

// next initialize another matrix with 1 or -1 based on the random
// value and whether it is above or below a given threshold.
  c1 = ( int * ) malloc ( m * n * sizeof ( int ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      // the threshold over which the cell is initialized to 1
      // and under or equal to which the cell is initialzed to -1
      if ( r[i+j*m] <= thresh )
      {
        c1[i+j*m] = -1;
      }
      else
      {
        c1[i+j*m] = +1;
      }
    }
  }
  free ( r );

  return c1;
}

/*****************************************************************************/

void dbl_mat_uniform_01 ( int m, int n, long unsigned int seed, double r[], int verbose  )
/*****************************************************************************/
/*
  Fills r, which is a'flattened' array of doubles that can be envisioned as
  a matrix of n rows and m columns, with random double values between 0 and 1.0.

*/
{
  // loop variables
  int i;
  int j;

  double randN;

  trng::yarn2 RNengine1; // try yarn2 for number generation
  RNengine1.seed(seed); 

  // special distribution that generates double numbers between 0 and 1.0
  trng::uniform01_dist<> uni;  

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      double randN = uni(RNengine1);
      r[i+j*m] =  randN; 
    }
  }

  if (verbose) {
    printf("sequential rand values in r:\n"); 
    for ( j = 0; j < n; j++ ) {
      for ( i = 0; i < m; i++ ) {
        printf("%lf ", r[i+j*m]);   
      }
      printf("\n"); 
    }
  }

  return;
}

/******************************************************************************/
/*
  Purpose: to generate a matrix of random numbers in parallel
*/
void parallel_dbl_mat_uniform_01( int m, int n, long unsigned int seed, double r[], int verbose, int nThreads  )
{
  // loop variables
  int i;
  int j;

  double randN;
  int rank;

  omp_set_num_threads(nThreads);

  #pragma omp parallel default(none) \
  private(j, i, randN, rank) shared(r, m, n, nThreads, seed)
  {
    trng::yarn2 RNengine1; // try yarn2 for number generation
    RNengine1.seed(seed); 

    rank = omp_get_thread_num();
    
     // special distribution that generates double numbers between 0 and 1.0
    trng::uniform01_dist<> uni;
    
    // simple leapfrog to assign random numbers to threads
    if (nThreads > 1)   RNengine1.split(nThreads, rank);
  
  #pragma omp for
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        randN = uni(RNengine1);
        r[i+j*m] =  randN; 
      }
      
    } 
  }
  if (verbose) {
    printf("Number of threads: %d\n", nThreads);
    printf("parallel rand values in r:\n"); 
    for ( j = 0; j < n; j++ ) {
      for ( i = 0; i < m; i++ ) {
        printf("%lf ", r[i+j*m]);   
      }
      printf("\n"); 
    }
  }
}
/******************************************************************************/

