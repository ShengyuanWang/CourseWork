/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * This software has been changed for educational purposes by:
 * Libby Shoop, Macalester College,   March, 2022
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include "helper_cuda.h"

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

// !!!!!! NOTE:
//        NVIDIA refers to these functions prefaced with __global__ 
//        as 'kernel' functions that run on the GPU 'device'.
//        All threads run this code, each having different values of:
//        gridDim.x (also y and z)
//        blockDim.x (also y and z)
//        blockIdx.x (also y and z)
//        threadIdx.x (also y and z)
// !!!!!!
__global__ void test1DGridKernel(int val) {
  // 1D grid of 1D blocks of threads means we consider horizontal direction
  int blocksPerGrid_horizontal = gridDim.x;
  int threadsPerBlock_horizontal = blockDim.x;
  
  // programmer can assign a unique block number for thread running this code
  int gridBlockNumber = blockIdx.x;
  // programmer can assign a unique thread number for the thread running this code
  int threadNumber = (gridBlockNumber * threadsPerBlock_horizontal) + threadIdx.x;

  // 
  printf("[b%d of %d, t%d]:\tValue is:%d\n", gridBlockNumber, blocksPerGrid_horizontal, threadNumber, val);
}

__global__ void test2DGridKernel(int val) {
  // 2D grid of 2D blocks of threads means we consider horizontal and vertical direction
  int blocksPerGrid_horizontal = gridDim.x;
  int blocksPerGrid_vertical = gridDim.y;
  int totalBlocks = blocksPerGrid_horizontal * blocksPerGrid_vertical;

  int threadsPerBlock_horizontal = blockDim.x;
  int threadsPerBlock_vertical = blockDim.y;
  int threadsPerBlock = threadsPerBlock_horizontal * threadsPerBlock_vertical;
  
  // programmer can assign a unique block number for thread running this code
  int gridBlockNumber = blockIdx.x + (blocksPerGrid_horizontal * blockIdx.y);
                          
  // programmer can assign a unique thread number for the thread running this code
  int threadNumber = (gridBlockNumber * threadsPerBlock) + 
                     (threadIdx.y * threadsPerBlock_horizontal) + threadIdx.x;


  // 
  printf("[b%d of %d, t%d]:\tValue is:%d\n", gridBlockNumber, totalBlocks, threadNumber, val);
}

// This function is designed to handle assigning thread numbers for
// 1D or 2D grids of blocks that are 1, 2, or 3 dimensions.
// It is therefore generic for any of these combinations.
// In CUDA we don't use 3D grids very often (just blocks), so this
// was left out of this example. 
__global__ void test2DGrid3DBlockKernel(int val) {
  // Here we will use the CUDA dim values
  // assume a 2D grid (this works with 1D also)
  int gridBlockNumber = blockIdx.y * gridDim.x + blockIdx.x;

  // assume a 3D block of threads (works with 1D and 2D also)
  int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  int threadNumberInBlock = threadIdx.z * blockDim.x * blockDim.y + 
                      threadIdx.y * blockDim.x + threadIdx.x;
  int threadNumber = (gridBlockNumber * threadsPerBlock ) + threadNumberInBlock;
 
  // 
  printf("[b%d, t%d, overall threadNum:%d]:\tValue is:%d\n", gridBlockNumber, threadNumberInBlock, threadNumber, val);
}

int main(int argc, char **argv) {
  
  getDeviceInformation();   // shows how many SMs on our device

  printf("\nprintf() is called in device code on GPU. Output:\n\n");

  //////////////////////////////////////////////////////////////
  //
  //    Each block that you specify maps to an SM.
  //////////////////////////////////////////////////////////////

  printf("Example 1. Host calls: 1 block of 32 threads\n");
  // 1 block of 32 threads goes to 1 SM 
  test1DGridKernel<<<1, 32>>>(10);

  cudaDeviceSynchronize();         // comment and re-make and run

  printf("Example 2. Host calls: 2 blocks of 32 threads\n");
  // 2 blocks to 2 SMs, 32 threads each
  test1DGridKernel<<<2, 32>>>(20);

  cudaDeviceSynchronize();
  printf("Example 3. Host calls: 2x2 grid of 4x2 blocks of 8 threads each\n");

  // Kernel configuration, where a two-dimensional grid and
  // two-dimensional blocks are configured.
  dim3 dimGrid2D(2, 2);                      // 2x2 = 4 blocks
  dim3 dimBlock2D(4, 2);                  // 4x2 = 8 threads per block
  test2DGridKernel<<<dimGrid2D, dimBlock2D>>>(30);

  cudaDeviceSynchronize();

  printf("Example 4. Host code: 2x2 grid of 2x2x2 blocks of 8 threads each\n");
  // Kernel configuration, where a two-dimensional grid and
  // three-dimensional blocks are configured.
  dim3 dimGrid(2, 2);                      // 2x2 = 4 blocks
  dim3 dimBlock(2, 2, 2);                  // 2x2x2 = 8 threads per block
  test2DGrid3DBlockKernel<<<dimGrid, dimBlock>>>(40);

  cudaDeviceSynchronize();

  // Note that this function works the same as the test1DGridKernel function
  printf("Host calls: 2 blocks of 32 threads\n");
  test2DGrid3DBlockKernel<<<2, 32>>>(50);

  cudaDeviceSynchronize();

  return 0;
}