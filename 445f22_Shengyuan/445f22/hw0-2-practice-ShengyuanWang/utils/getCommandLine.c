// #include <iostream>
// #include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
// #include <cstring.h>
#include <ctype.h>
#include "getCommandLine.h"


void getArguments(int argc, char *argv[],
                  int * verbose, int *numThreads)
{
  
  char *numThreads_value;  // number of threads when entered at command line is a string of chars
  
  int c;        // result from getopt calls

  // for verbose printing output 
  int verbose_flag = 0;
  
  // The : after a character means a value is expected, such as
  //      ./hello -t 4
  // No colon means it is simply a flag with no associated value, such as
  //      ./hello -v
  //      ./hello -h
  while ((c = getopt (argc, argv, "t:vh")) != -1) {

    switch (c)
      {
      // character string entered after the -t needs to be a number
      case 't':
        if (isNumber(optarg)) {
          numThreads_value = optarg;
          *numThreads = atoi(numThreads_value);
        } else {
          exitWithError(c, argv);
        } 
        break;

      // If the -v is encountered, then we change the flag
      case 'v':
        verbose_flag = 1;
        *verbose = verbose_flag;
        break;
      
      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0]);
        exit(0);  
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
void Usage(char *program) {
  fprintf(stderr, "Usage: %s [-h] [-v] [-t numThreads]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -v provides verbose output.\n");
  fprintf(stderr, "   -t indicates the number of threads to use. Default is 1 without this flag.\n");
}
