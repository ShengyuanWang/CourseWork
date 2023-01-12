/******************************************************************************
* FILE: omp_hello.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the main thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The main thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number. Others are used to demonstrate use of a verbose flag.
* AUTHOR: Blaise Barney  5/99
* LAST REVISED: 04/06/05
* LAST UPDATED by Libby Shoop for educational purposes:   08/29/22
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "./utils/getCommandLine.h"

int main (int argc, char *argv[]) {

  int nthreads =1;  //default value for number of threads is always a good plan
  int tid = 0;      // as we will see, with 1 thread, our tid is by convention 0
  int verbose = 0;  // determines if more information should be printed.

// TODO 1: set the value of nthreads to some value less than 32,
// by uncommenting the following line and trying different values.
// nthreads = 4;
//
// !! remember to save this file and recompile each time 
// you change the value

// TODO 2: Comment out the line above that 'hard codes' the value 
//        of nthreads and instead work on getting the value of 
//        nthreads from the command line by using the function 
//        called getArguments() found in utils/getCommandLine.c.
//        Open that file and study it carefully. 
//        Then uncomment this call of it:
getArguments(argc, argv, &verbose, &nthreads);

// Set the number of threads to use.
// This is a standard practice that you should use to ensure that
// your code is doing what you intend:
//  Take in the number of threads/processes from the command line
//  using a function like getArguments(), then set the value the use input.
omp_set_num_threads(nthreads);

// Get some details if verbose mode chosen.
// This is a demonstration of the verbose flag programming tehnique,
// which can be useful for debugging or creating more detailed results.
if (verbose) {
  if (omp_get_nested()){
    printf("Nested parallelism is enabled.\n", tid);
  } else {
    printf("Nested parallelism is not enabled.\n", tid);
  }

  // Note how we can check whther we have set the number of threads
  // to be used as a default. We did this with omp_set_num_threads() above.
  int max_num_threads = omp_get_max_threads();
  printf("Maximum number of threads is %d.\n", max_num_threads);
}

printf("Just before forking, still in thread %d.\n During fork:\n\n", omp_get_thread_num());

/* Fork a team of threads giving them their own copies of variables
  nthreads and tid. Note that they share one copy of the variable 
  called verbose.
   */
#pragma omp parallel private(nthreads, tid) shared(verbose)\
        default(none)
  {

    /* Obtain thread number */
    tid = omp_get_thread_num();
    printf("Hello World from thread = %d\n", tid);

    /* Only main thread does this */
    if (tid == 0) {
      nthreads = omp_get_num_threads();
      printf("Number of threads = %d\n", nthreads);
    }

    /* Provide more information from some other functions */
    if (verbose) {
      if (omp_in_parallel()){
        printf("Thread %d in a parallel region.\n", tid);
      }
      
    }

  }  /* All threads join main thread and disband */

  printf("After threads join back together...\n\n");
  tid = omp_get_thread_num();
  printf("Hello again from thread = %d at end of program.\n", tid);

  return 0;   // main returns 0 if no errors
}
