// #include <iostream>
// #include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
// #include <cstring.h>
#include <ctype.h>
#include "getCommandLine.h"

void getArgumentsSeq(int argc, char *argv[],
                  int * verbose, int * N, int * experimentPrint)
{
  
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:vhe:")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n
    switch (c)
      {
      // character string entered after the -N needs to be a number
      case 'n':
        if (isNumber(optarg)) {
          *N = atoi(optarg);
        } else {
          exitWithError(c, argv);
        } 
        break;

      // If the -v is encountered, then we change the flag
      case 'v':
        *verbose = 1;
        break;
      
      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0]);
        exit(0);  
        break;

      // If the -e is encountered, then we change the flag
      case 'e':
        if (isNumber(optarg)) {
          *experimentPrint = atoi(optarg);
        } else {
          exitWithError(c, argv);
        } 
        break;

      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (
            (optopt == 'v') ||
            (optopt == 't')
           ) 
        {
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        } else if (isprint (optopt)) {
          fprintf (stderr, "Unknown option `-%c'.\n", optopt);
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        } else {
          fprintf (stderr,
                   "Unknown non-printable option character `\\x%x'.\n",
                   optopt);
          UsageSeq(argv[0]);
          exit(EXIT_FAILURE);
        }
        break;
      
      }
  }
}

void getArguments(int argc, char *argv[],
                  int * verbose, int * N, int *numThreads, int*experimentPrint)
{
  
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected, such as
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:t:vhe:")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n or -t
    switch (c)
      {
      // character string entered after the -n needs to be a number
      case 'n':
        if (isNumber(optarg)) {
          *N = atoi(optarg);
        } else {
          exitWithError(c, argv);
        } 
        break;
      // character string entered after the -t needs to be a number
      case 't':
        if (isNumber(optarg)) {
          *numThreads = atoi(optarg);
        } else {
          exitWithError(c, argv);
        } 
        break;

      // If the -v is encountered, then we change the flag
      case 'v':
        *verbose = 1;
        break;
      
      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0]);
        exit(0);  
        break;

      case 'e':
        if (isNumber(optarg)) {
          *experimentPrint = atoi(optarg);
        } else {
          exitWithError(c, argv);
        } 
        break;


      case ':':
        printf("Missing arg for %c\n", optopt);
        Usage(argv[0]);
        exit(EXIT_FAILURE);
        break;

      case '?':
        if (
            (optopt == 'v') ||
            (optopt == 't')
           ) 
        {
          Usage(argv[0]);
          exit(EXIT_FAILURE);
        } else if (isprint (optopt)) {
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
void exitWithError(char cmdFlag, char ** argv) {
  fprintf(stderr, "Option -%c needs a number value\n", cmdFlag);
  Usage(argv[0]);
  exit(EXIT_FAILURE);
}

// All useful C/C++ programs with command line arguments produce a
// Usage string to the screen when there is an issue or when help is requested.
//sequential version
void UsageSeq(char *program) {
  fprintf(stderr, "Usage: %s [-h] [-v] [-n numElements] [-e printPart]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -v provides verbose output.\n");
  fprintf(stderr, "   -n indicates the number of elements to create that will be sorted\n");
  fprintf(stderr, "   -e indicates which part of the experiment results to print: 0(all results)\t1(generate part)\t2(sort part)");
}
// parallel threaded version
void Usage(char *program) {
  fprintf(stderr, "Usage: %s [-h] [-v] [-n numElements] [-t numThreads] [-e printPart]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -v provides verbose output.\n");
  fprintf(stderr, "   -n indicates the number of elements to create that will be sorted\n");
  fprintf(stderr, "   -t indicates the number of threads to use. Default is 1 without this flag.\n");
  fprintf(stderr, "   -e indicates which part of the experiment results to print: 0(all results)\t1(generate part)\t2(sort part)");
}
