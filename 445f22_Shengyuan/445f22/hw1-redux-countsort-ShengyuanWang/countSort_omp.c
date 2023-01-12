/*
 * Count sort example
 *
 * Applicability: When the keys are integers drawn from a
 * small range, such as 8-bit numbers, and therefore contain
 * many duplicates.
 *
 * Author: Libby Shoop
 *        with inspiration from:
 *        https://www8.cs.umu.se/kurser/5DV011/VT13/F8.pdf
 *
 * compile with the Makefile and make
 *    
 *
 * Usage example (designed to take in various problem sizes for input string):
 *     ./countSort_seq -n 8388608
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <omp.h>
#include <trng/yarn2.hpp>
#include <trng/lcg64.hpp>
#include <trng/uniform_dist.hpp>
#include "./utils/getCommandLine.h"

#define ASCII_START 32
#define ASCII_END 126
#define COUNTS_SIZE ASCII_END+1

// functions in this file
char* generateRandomString(int size, int nthreads);
void countEachLetter(int * counts, char * input, int N);
void createSortedStr(int * counts, char * output, int N);

int main(int argc, char *argv[]) {

    // sample values for profiling
    // int N = 40;
    // int N = 8376000; // this should work for 512M machine
    // int N = 8388608;  // 2^20
    // int N = 16777216; // 2^24
    // int N = 67108864;  // 2^26
    // int N = 1073741824;  // 2^30 takes 6 seconds to generate the string with rand_r
    // 2^31, 2147483648, fails (seg fault; out of memory), but close to it works, like
    //  2147483000
    // 2147483616 seems to be the max

  int nthreads = 1;  //default value for number of threads is always a good plan
  int verbose = 0;  // determines if more information should be printed.
  int experimentPrint = 0;
  float generate_time, sort_time;
  int N = 40;  // size of input string; default to small number for debugging

  getArguments(argc, argv, &verbose, &N, &nthreads, &experimentPrint);  // should exit if -n has no value
  //debug
  if (verbose) {
    printf("number of chars in input: %d\n", N);
    printf("number of threads is : %d\n", nthreads);
  }
  /****** Initialization ********/
  /*** OMP ***/
  omp_set_num_threads(nthreads);
  /*** OMP ***/


  // for timing using the openMP timing function
  double start, end;

  // for profiling, time each section of code, observing how much
  // each one takes.
  start = omp_get_wtime();    
  // create fake sample data to sort
  char * input = generateRandomString(N, nthreads);
  end = omp_get_wtime();
  generate_time = end - start;
  if (experimentPrint == 1) {
    printf("%lf\t", end - start);
  }
  if (experimentPrint == 0) {
    printf("generate input: %f seconds\n", end - start);
  }

  // char output[N+1];  //eliminate this so can do larger problems

  //debug
  if (verbose) {
    printf("input: %s\n", input);
  }

  int counts[COUNTS_SIZE] = {0}; // indexes 0 - 31 should have zero counts.
                                // we'll include them for simplicity
  start = omp_get_wtime();
  countEachLetter(counts, input, N);
  end = omp_get_wtime();
  sort_time = end - start;
  if (experimentPrint == 0) {
    printf("generate counts: %f seconds\n", end - start);
  }
  

  start = omp_get_wtime();
  createSortedStr(counts, input, N);   //put the result back into the input
  end = omp_get_wtime();
  sort_time = sort_time + end - start;
  if (experimentPrint == 2) {
    printf("%lf\t", sort_time);
  }
  if (experimentPrint == 0) {
    printf("generate output: %f seconds\n", end - start);
  }
  

  //debug
  if (verbose) {
    printf("output: %s\n", input);
  }

  free(input);
  return 0;
}
//////////////////// end of main /////////////////////////////////////

//  Creation of string of random printable chars.
//  Normally this type of string would come from a data source.
//  Here we generate them for convenience.
//
char* generateRandomString(int size, int nThreads) {
    // long unsigned int seed = 123456;
    // fixed seed generates same string every time FOR DEBUG ONLY
    // seed = 1403297347; 
    // if you do the following the string changes every time you run it
    long unsigned int seed = time(NULL);
    int tmp = 0;
    char *res = (char *)malloc(size + 1);
    unsigned int tid; // thread id when forking threads
    unsigned min = ASCII_START;
    unsigned max = ASCII_END;
    
    int i;
    char nextRandVal;
    // fork threads to generate random character in parallel
    #pragma omp parallel default(none) \
    private(tid, i, nextRandVal) \
    shared(size, nThreads, res, min, max, seed) 
    {
      trng::lcg64 rand; 
      trng::uniform_dist<> uniform(min, max+1);
      rand.seed(seed);
      tid = omp_get_thread_num();
      if (nThreads > 1){
        // thread will be substream tid from numThread separate substreams
        rand.split((unsigned)nThreads, tid);
      }
      // openMP splits the work
      #pragma omp for
      for (i=0; i<size; i++) {
        nextRandVal = uniform(rand);
        res[i] = nextRandVal;
        // printf("t%2d(%2d):%c \n", tid, i, nextRandVal);
      }

    }

    res[size] = '\0';  //so it is a string we can print when debugging
    return res;
}

// The input called counts is designed so that an index into it
// represents tha code of an ascii character. The array input holds
// such ascii characters.
// More generally, the indices into counts are the set of keys we
// are sorting.
// We will first count how many times we see each key.
void countEachLetter(int * counts, char * input, int N) {
    // Count occurences of each key, which are characters in this case
    #pragma omp parallel for default(none) \
    shared(N, input)  \
    reduction(+:counts[:COUNTS_SIZE])
    for ( int k = 0; k < N ; ++ k ) {
        counts [ input [ k ] ] += 1;
        int id = omp_get_thread_num();
        // printf("Thread %d performed iteration %d\n", 
        //          id, k);
    }
    // printf("Finish this part.");
}

// Create the sorted string by adding chars in order to an output, using
// the count to create the correct number of them.
//
// We choose to save space by instead placing the final
// result back into input. See use above.
void createSortedStr(int * counts, char * output, int N) {
    // prefix sum array
    int prefixArr[COUNTS_SIZE] = {0};

    for ( int i = 0; i < COUNTS_SIZE; i++) {
      if (i == 0) {
        prefixArr[0] = counts[0];
      } else {
        prefixArr[i] = prefixArr[i-1] + counts[i];
      }
    }
    // Construct output array from counts
    char v;
    int start, end;
  
    #pragma omp parallel for default(none) \
    private(v, start, end)\
    shared(output, prefixArr)
    for ( v = 0; v <= ASCII_END; ++ v ) {
      if (v == 0) {
        start = 0;
      } else {
        start = prefixArr[v-1];
      }
      end = prefixArr[v];
      for ( int k = start; k < end; ++ k ) {
        output[ k ] = v ;
      }
    }
    output[N+1] = '\0';   // so it is a string we could print for debugging
}
