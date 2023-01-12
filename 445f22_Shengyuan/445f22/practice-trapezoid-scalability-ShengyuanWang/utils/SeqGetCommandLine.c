
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include "SeqGetCommandLine.h"

void getArguments(int argc, char *argv[],
                  unsigned long int * N, int * verbose, int *experimentPrint)
{
  int c;        // result from getopt calls
  
  // The : after a character means a value is expected
  // No colon means it is simply a flag with no associated value
  while ((c = getopt (argc, argv, "n:hve")) != -1) {

// getopt implicitly sets a value to a char * (string) called optarg
// to what the user typed after -n
    switch (c)
      {
      // character string entered after the -N needs to be a number
      case 'n':
        if (isNumber(optarg)) {
          char *ptr; // not used
          *N = strtoul(optarg, &ptr, 10);
        } else {
          exitWithError(c, argv); 
        } 
        break;

      // If the -h is encountered, then we provide usage
      case 'h':
        Usage(argv[0]);
        exit(0);  
        break;

      // If the -v is encountered, then we verbose print
      case 'v':
        *verbose = 1;
        break;

      // If the -e is encountered, then we print for experiments
      case 'e':
        *experimentPrint = 1;
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

void Usage(char *program) {
  fprintf(stderr, "This program computes an integral using the trapezoidal rule\n.");
  fprintf(stderr, "Usage: %s [-h] [-v] [-n numTrapezoids]\n", program);
  fprintf(stderr, "   -h shows this message and exits.\n");
  fprintf(stderr, "   -n indicates the number of trapezoinds to use.\n");
  fprintf(stderr, "   -v indicates verbose printing.\n");
  fprintf(stderr, "   -e indicates printing for running experiments.\n");
}
