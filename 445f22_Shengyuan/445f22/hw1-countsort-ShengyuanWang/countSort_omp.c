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
 * compile with:
 *     gcc -std=gnu99 -o countSort_seq countSort_seq.c
 *
 * Usage example (designed to take in various problem sizes for input string):
 *     ./countSort -n 8388608
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

#include "seq_time.h"  // Libby's timing function that is similar to omp style
//new added functions
#include <omp.h>
#include <time.h>
#include <trng/yarn2.hpp>
#include <trng/lcg64.hpp>
#include <trng/uniform_dist.hpp>
#define ASCII_START 32
#define ASCII_END 126
#define COUNTS_SIZE ASCII_END+1

// functions in this file
void getArguments(int argc, char *argv[], int * N, int * nThreads);
char* generateRandomString(int size);
void countEachLetter(int * counts, char * input, int N);
void createSortedStr(int * counts, char * output, int N);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char **argv, int isPar);

int main(int argc, char *argv[]) {

    // simple hard-code for profiling
    // int N = 40;
    // int N = 8376000; // this should work for 512M machine
    // int N = 8388608;  // 2^20
    // int N = 16777216; // 2^24
    // int N = 67108864;  // 2^26
    // int N = 1073741824;  // 2^30 takes 6 seconds to generate the string
    // 2^31, 2147483648, fails (seg fault; out of memory), but close to it works, like
    //  2147483000

    int N;  // size of input string

    // change 1: add nThreads
    int nThreads = 1; // default number of Threads

    getArguments(argc, argv, &N, &nThreads);  // should exit if -n has no value
    //debug
    //printf("number of chars in input: %d\n", N);
    double start, end;

    // start = c_get_wtime();    // see seq_time.h if you want details
    start = omp_get_wtime();
    // create fake sample data to sort
    char * input = generateRandomString(N);
    end = omp_get_wtime();
    // end = c_get_wtime();
    printf("generate input: %f seconds\n", end - start);

    // char output[N+1];  //eliminate this so can do larger problems

    //debug
    //printf("input: %s\n", input);

    int counts[COUNTS_SIZE] = {0}; // indexes 0 - 31 should have zero counts.
                                 // we'll include them for simplicity
    // start = c_get_wtime();
    start = omp_get_wtime();
    countEachLetter(counts, input, N);
    end = omp_get_wtime();
    // end = c_get_wtime();
    printf("generate counts: %f seconds\n", end - start);

    // start = c_get_wtime();
    start = omp_get_wtime();
    // createSortedStr(counts, output);
    createSortedStr(counts, input, N);   //put the result back into the input
    end = omp_get_wtime();
    // end = c_get_wtime();
    printf("generate output: %f seconds\n", end - start);

    //debug
    //printf("output: %s\n", input);

    free(input);
    return 0;
}

// process command line looking for the number of characters
// in the input string as our 'problem size'. Set the value of N
// of N to that number or generate error if not provided.
//   see:
// https://www.gnu.org/software/libc/manual/html_node/Example-of-Getopt.html#Example-of-Getopt
//
void getArguments(int argc, char *argv[], int * N, int * nThreads) {
    char *nvalue;
    int c;        // result from getopt calls
    int nflag = 0;

    while ((c = getopt (argc, argv, "n:t:")) != -1) {
      switch (c)
        {
        case 'n':
          nflag = 1;
          nvalue = optarg;
          *N = atoi(nvalue);
          break;
        //change 1: add function -t
        // character string entered after -t needs to be a number 
        case 't':
          if (isNumber(optarg)) {
            *nThreads = atoi(optarg);
          } else {
            exitWithError(c, argv);
          }


        case '?':
          if (optopt == 'n') {
            fprintf (stderr, "Option -%c requires an argument.\n", optopt);
          } else if (isprint (optopt)) {
            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          } else {
            fprintf (stderr,
                     "Unknown option character `\\x%x'.\n",
                     optopt);
            exit(EXIT_FAILURE);
          }

        }
    }
    if (nflag == 0) {
      fprintf(stderr, "Usage: %s -n size\n", argv[0]);
      exit(EXIT_FAILURE);
    }
}

//  Creation of string of random printable chars.
//  Normally this type of string would come from a data source.
//  Here we generate them for convenience.
//
char* generateRandomString(int size, int nThreads) {
    unsigned int seed = time(NULL); // for default seed the computer clock
    int i;
    char *res = malloc(size + 1); // the output string
    unsigned int tid; // thread id when forking threads
    unsigned min = ASCII_START;
    unsigned max = ASCII_END;
    // fork threads to generate random character in parallel
    #pragma omp parallel default(none) \
    private(tid) \
    shared(size, nThreads, min, max, seed) \
    {
      trng::lcg64 rand; 
      rand.seed(seed);
      trng::uniform_dist<> uniform(min, max);
      tid = omp_get_thread_num();
      if (nThreads > 1){
        // thread will be substream tid from numThread separate substreams
        rand.split((unsigned)nThreads, tid);
      }
      int nextRandVal;
      int tmp;
      // openMP splits the work
      #pragma omp for
      for (tmp=0; tmp<size; tmp++) {
        nextRandVal = uniform(rand);
        res[tmp] = (char)nextRandVal;
        printf("t%2d(%2d):%2d \n", tid, tmp, nextRandVal);
      }

    }

    // for(i = 0; i < size; i++) {
    //     res[i] = (char) ((rand_r(&seed) % (ASCII_END-ASCII_START)) + ASCII_START);
    // }

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
    for ( int k = 0; k < N ; ++ k ) {
        counts [ input [ k ] ] += 1;
    }
}

// Create the sorted string by adding chars in order to an output, using
// the count to create the correct number of them.
//
// We choose to save space by instead placing the final
// result back into input. See use above.
void createSortedStr(int * counts, char * output, int N) {
    // Construct output array from counts
    int r = 0;
    for ( char v = 0; v <= ASCII_END; ++ v ) {
      for ( int k = 0; k < counts[v]; ++ k ) {
        output [ r ++ ] = v ;
      }
    }
    output[N+1] = '\0';   // so it is a string we could print for debugging
}

// Check a string as a number containing all digits
int isNumber(char s[])
{
  for (int i = 0; s[i]!= '\0'; i++)
  {
      if (isdigit(s[i]) == 0)
            return 0;
  }
  
  return 1;
}

// Called when isNumber() fails
void exitWithError(char cmdFlag, char ** argv, int isPar) {
  fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
  if (isPar) {
    UsagePar(argv[0]);
  } else {
    UsageSeq(argv[0]);
  }
  exit(EXIT_FAILURE);
}