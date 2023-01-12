// functions in getCommandLine.c file
void getArgumentsSeq(int argc, char *argv[],
                  int * verbose, int * N, int *experimentPrint);
void getArguments(int argc, char *argv[],
                  int * verbose, int * N, int *numThreads, int *experimentPrint);
int isNumber(char s[]);
void exitWithError(char cmdFlag, char ** argv);
void UsageSeq(char *program);
void Usage(char *program);
