#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
// #include <cstring.h>
#include "getCommandLine.hpp"

// void getArguments(int argc, char *argv[], int * nThreads, int * m, int * n, int * iterations, double * thresh, int * display, int *verbose)
void getArguments(int argc, char *argv[], int * m, int * n, 
                  int * iterations, double * thresh, 
                  int * useConstantSeed, int * verbose, int * numThreads)
{
  // arguments expected that have default values in the code: 
  // m = 10; n = 10; iterations = 15; thresh = 0.50;
  // m, n i, p
  //
  //char *tvalue;  // number of threads
  char *nvalue;       // number of rows
  char *mvalue;      // number of columns
  char *iters_value; // number of iterations
  char *probThreshold_value;  // probability threshold
  
  int c;        // result from getopt calls

  double converted;   // for floating point threshold value

// for threads later
// while ((c = getopt (argc, argv, "t:m:n:i:p:dv")) != -1) { 
  while ((c = getopt (argc, argv, "m:n:i:p:cvt:")) != -1) {

    switch (c)
      {
      case 't':
        if (isNumber(optarg)) {
          *numThreads = atoi(optarg);
        } else {
          exitWithError(c, argv);
        }
       break;

      case 'm':
        if (isNumber(optarg)) {
          mvalue = optarg;
          *m = atoi(mvalue);
        } else {
          exitWithError(c, argv);
        }
        break;
    
      case 'n':
        if (isNumber(optarg)) {
          nvalue = optarg;
          *n = atoi(nvalue);
        } else {
          exitWithError(c, argv);
        }
        break;

      case 'i':
        if (isNumber(optarg)) {
          iters_value = optarg;
          *iterations = atoi(iters_value);
        } else {
          exitWithError(c, argv);
        } 
        break;

      case 'p':
        probThreshold_value = optarg;
        converted = strtod(probThreshold_value, NULL);
        if (converted != 0 ) {
          *thresh = converted;
        } else {
          exitWithError(c, argv);
        } 
        break;

      case 'c':
        *useConstantSeed = 1;
        break;

      case 'v':
        *verbose = 1;
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (isprint (optopt)) {
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        } else {
          fprintf (stderr,
                   "Unknown non-printable option character `\\x%x'.\n",
                   optopt);
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        }
        break;
      
      }
  }
}

int isNumber(char s[])
{
    for (int i = 0; s[i]!= '\0'; i++)
    {
        if (isdigit(s[i]) == 0)
              return 0;
    }
    
    return 1;
}

void exitWithError(char cmdFlag, char ** argv) {
  fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
  Usage(argv[0]);
  exit(EXIT_FAILURE);
}

void Usage(char *program) {
    // fprintf(stderr, "Usage: %s [-t numThreads] [-m rows] [-n cols] [-i iterations] [-p threshold] [-d]\n", program);
  fprintf(stderr, "Usage: %s [-n rows] [-m cols] [-i iterations] [-p threshold] [-v] [-c] [-t num threads]\n", program);
}
