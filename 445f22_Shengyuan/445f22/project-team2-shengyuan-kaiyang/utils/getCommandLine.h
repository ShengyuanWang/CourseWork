// functions in getCommandLine.c file
void getArguments(int argc, char *argv[], int * N, int *numThreads, int * verbose, int *experiment);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void Usage(char *program);
