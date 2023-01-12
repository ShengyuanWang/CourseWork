/*
 *  Use cuda functions to indicate device information.
 */
// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include "helper_cuda.h"

int main(int argc, char **argv) {
  
  // shows how many SMs on our device, among other things
  getDeviceInformation();   

  return 0;
}