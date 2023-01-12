void getArguments(int argc, char *argv[],
                  unsigned long int * N, int * verbose, int *experimentPrint, 
                  int * numThreads);

int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void Usage(char *program);
