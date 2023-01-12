/*
 *  Simplified functions for checking info about a GPU device.
 */

 #include <stdio.h>
 #include <cuda.h>

// Find out info about a GPU.
// See this page for list of all the values we can "query" for:
// https://rdrr.io/github/duncantl/RCUDA/man/cudaDeviceGetAttribute.html
//
void getDeviceInformation() {
  int devId;            // the number assigned to the GPU
  int threadsPerBlock;  // maximum threads available per block
  int memSize;          // shared mem in each streaming multiprocessor (SM)
  int numProcs;         // number of SMs
  // !!!!! NOTES: 
  //     see figure 6.1 of Pacheco & Malensek textbook for architecture of GPU.
  //

  struct cudaDeviceProp props;

  cudaGetDevice(&devId);

  cudaDeviceGetAttribute(&threadsPerBlock, 
    cudaDevAttrMaxThreadsPerBlock, devId);
  cudaDeviceGetAttribute(&memSize, 
    cudaDevAttrMaxSharedMemoryPerBlock, devId);
  cudaDeviceGetAttribute(&numProcs,
    cudaDevAttrMultiProcessorCount, devId);
  
  cudaGetDeviceProperties(&props, devId);
  printf("Device %d: \"%s\" with Compute %d.%d capability\n", devId, props.name,
         props.major, props.minor);

  printf("GPU device maximum available threads per block: %d\n", threadsPerBlock);
  printf("GPU device shared memory per block of threads on an SM: %d bytes\n", memSize);
  printf("GPU device total number of streaming multiprocessors: %d\n", numProcs);

}