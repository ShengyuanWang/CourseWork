//
// Demonstration using a single 1D grid and 1D block size
//
#include <math.h>   // ceil function
#include <stdio.h>  // printf

#include <cuda.h>
// helper functions and utilities 
#include "helper_add.h"

// Kernel function based on 1D grid of 1D blocks of threads
// In this version, thread number is:
//  its block number in the grid (blockIdx.x) times 
// the threads per block plus which thread it is in that block.
//
// This thread id is then the index into the 1D array of floats.
// This represents the simplest type of mapping:
// Each thread takes care of one element of the result
__global__ void vecAdd(float *x, float *y, int n)
{
    // Get our global thread ID
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        y[id] = x[id] + y[id];
}


int main(int argc, char **argv)
{
  printf("Vector addition using managed memory.\n");
  // Set up size of arrays for vectors
  // int N = 1<<20;
  // same value, shown as multiple of 1024, 
  // which is divisible by 32 (size of the SPs on SM)
  int N = 1024*1024;   
  printf("size (N) of 1D arrays are: %d\n\n", N);
  // host vectors
  float *x, *y;

   // Size, in bytes, of each vector; just use below
  //size_t bytes = N*sizeof(float);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  initialize(x, y, N);  // set values in each vector

  // Number of threads in each thread block
  int blockSize = 1024;
 
  // Number of thread blocks in grid needs to be based on array size
  int gridSize = (int)ceil((float)N/blockSize);
 
  printf("add vectors on device\n");
  // Execute the kernel
  vecAdd<<<gridSize, blockSize>>>(x, y, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  checkForErrors(y, N);

  printf("execution complete\n");

  // free unified memory
  cudaFree(x);
  cudaFree(y);

  return 0;

}