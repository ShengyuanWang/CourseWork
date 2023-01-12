/*
 * Example of vector addition from this NVIDIA website:
 *   https://developer.nvidia.com/blog/even-easier-introduction-cuda/
 *
 * Array of floats x is added to array of floats y and the 
 * result is placed back in y
 */

#include <iostream>
#include <math.h>

// functions in this file
// simple command line argument conversion
void getArguments(int argc, char **argv, int *numBlocks, int *blockSize);
// array initialization
void initialize(float *x, float *y, int N);
// check that there was not an error in our code for adding vectors
void checkForErrors(float *y, int N);

// Kernel function to add the elements of two arrays
// This one is still sequential.
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}

// Parallel version that uses threads in the block.
//
//  If block size is 8, e.g.
//    thread 0 works on index 0, 8, 16, 24, etc. of each array
//    thread 1 works on index 1, 9, 17, 23, etc.
//    thread 2 works on index 2, 10, 18, 24, etc.
//
// This is mapping a 1D block of threads onto these 1D arrays.
__global__
void add_parallel_1block(int n, float *x, float *y)
{
  int index = threadIdx.x;    // which thread am I in the block?
  int stride = blockDim.x;    // threads per block
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

// In this version, thread number is its block number 
// in the grid (blockIdx.x) times 
// the threads per block plus which thread it is in that block.
//
// Then the 'stride' to the next element in the array goes forward
// by multiplying threads per block (blockDim.x) times 
// the number of blocks in the grid (gridDim.x).

__global__
void add_parallel_nblocks(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}


int main(int argc, char **argv)
{
  printf("This program lets us experiment with block sizes and\n");
  printf("number of threads per block to see it effect on running time.\n");
  printf("Usage:\n");
  printf("%s [num threads per block] [num blocks]\n", argv[0]);
  printf("\nwhere you can specify only the number of threads per block \n");
  printf("and the number of blocks will be calculated based on the size\n");
  printf("of the array, which is hard-coded to 1,046,576 elements.\n\n");

  // for timing using CUDA functions
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0.0;

  // Set up size of arrays
  // int N = 1<<20;
  int N = 1024*1024;   // same value, shown as multiple of 1024
  printf("size (N) of 1D array is: %d\n\n", N);
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  initialize(x, y, N);
  
///// Purely illustration of something you do not ordinarilly do:
  // Run kernel on all elements on the GPU sequentially

  cudaEventRecord(start);
  add<<<1, 1>>>(N, x, y);   // the kernel call
  cudaEventRecord(stop);
  milliseconds = 0.0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\nSequential time on one device thread: %f milliseconds\n", milliseconds);
  // Note that this type of call is still sequential on the device:
  // add<<<1, 256>>>(N, x, y);

// Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  checkForErrors(y, N);

///////////////////////////////////////////////////////////////////
// Now on block of 256 threads
  // re-initialize x and y arrays on the host
  initialize(x, y, N);

  int blockSize = 256;
  int numBlocks = 1;

  // Use the GPU in parallel with one block of threads.
  // Essentially using one SM for the block.
  cudaEventRecord(start);
  add_parallel_1block<<<1, 256>>>(N, x, y);   // the kernel call
  cudaEventRecord(stop);
  milliseconds = 0.0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("\nParallel time on %d block of %d threads: %f milliseconds\n", numBlocks, blockSize, milliseconds);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  checkForErrors(y, N);

///////////////////////////////////////////////////////////////////
  // Now use multiple blocks so that we use more than one SM.
  
  /////////              Option 1    7.8 ms when profiled
  // blockSize = 256;
  // Knowing the length of our array, determine the number of 
  // blocks of 256 threads we need to map each array element to a
  // thread in a block.
  // numBlocks = (N + blockSize - 1) / blockSize;
   
  //////////////////    Option 2  7.47 ms when profiled
  // blockSize = 1024;
  // numBlocks = (N + blockSize - 1) / blockSize;
  
  // re-initialize x and y arrays on the host
  initialize(x, y, N);

  ////////////////     Default option:   match our card's architecture
  ////////////////         6.5  to 7.08 ms when profiled
  // blockSize = 1024;  // our card has a max of this many threads per block
  // numBlocks = 40; // our card has 40 SMs

  blockSize = 256;
  numBlocks = 0;      // signifies we should calculate numBlocks

  getArguments(argc, argv, &numBlocks, &blockSize); //override
  // Knowing the length of our array, determine the number of 
  // blocks threads we need to map each array element to a
  // thread in a block.
  if (numBlocks == 0) {   //signifies we should calculate numBlocks
    numBlocks = (N + blockSize - 1) / blockSize;
  }

  printf("\n----------- number of %d-thread blocks: %d\n", blockSize, numBlocks);

  cudaEventRecord(start);
     // the kernel call
  add_parallel_nblocks<<<numBlocks, blockSize>>>(N, x, y);
  cudaEventRecord(stop);
  milliseconds = 0.0;
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Parallel time on %d block of %d threads: %f milliseconds\n", numBlocks, blockSize, milliseconds);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  checkForErrors(y, N);

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}

// To reset the arrays for each trial
void initialize(float *x, float *y, int N) {
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
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
