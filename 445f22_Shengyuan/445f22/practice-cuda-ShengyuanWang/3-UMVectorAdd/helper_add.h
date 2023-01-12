/*
 * Functions for setting up arrays, getting arguments
 */
#include <iostream>
 #include <cuda.h>

// To reset the arrays for each trial
void initialize(float *x, float *y, int N) {
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
}

// CPU version of add sequentially
void CPUadd(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// check whether the kernel functions worked as expected
void checkForErrors(float *y, int N) {
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
}
// simple argument gather for this simple example program
void getArguments(int argc, char **argv, int *numBlocks, int *blockSize) {

  if (argc == 3) {
    *numBlocks = atoi(argv[2]);
    *blockSize = atoi(argv[1]);
  } else if (argc == 2) {
    *blockSize = atoi(argv[1]);
  }

}
