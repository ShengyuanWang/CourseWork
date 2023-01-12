// functions in getCommandLine.c file
void getArguments(int argc, char *argv[], int * m, int * n, 
                  int * iterations, double * thresh, 
                  int * useConstantSeed, int * verbose, int * numThreads);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void Usage(char *program);