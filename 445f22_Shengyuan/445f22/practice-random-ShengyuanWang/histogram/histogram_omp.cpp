/*
 * Create random number sequence in parallel, placing each number
 * generated into a 'bin'.
 *
 * Usage:
 *  Print runtime: ./histogram_omp -t 16 -n 100000000 -i 15 -a 0 -b 100
 *  Print histogram data: add tag "-p"
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <omp.h>

#include <trng/lcg64_shift.hpp>
#include <trng/lcg64.hpp>
#include <trng/yarn2.hpp>
#include <trng/normal_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/exponential_dist.hpp>

// separate file for handling command line arguments
#include "./utils/getCommandLinePar.h"

using namespace std;   // for 'cout' style of printing in C++

///==============  main ======================
int main(int argc, char* argv[])
{
  int nThreads = 16;
  int N = 100000000;
  int numBins = 15;
  // use these to experiment with a distribution of doubles
  // double min = 0;
  // double max = 10000;
  int min = 0;
  int max = 10000;
  int print = 0;
  int useConstantSeed = 0;

  int index, rank, i;
  // double randN;    // if want to try double values
  int randN;
  double d_index;

  getArguments(argc, argv, &nThreads, &N, &numBins, &min, &max, &print, &useConstantSeed);
  //debug
  // printf("%d random values from %d to %d placed into %d bins\n", N, min, max, numBins);

  int hist_array [numBins] = {};

  // random numbers start from a seed value
  long unsigned int seed;  // note for trng this is long unsigned

  double start = omp_get_wtime();

#pragma omp parallel num_threads(nThreads) \
  shared(N, max, min, numBins, nThreads, seed, useConstantSeed) \
  private(i, d_index, index, rank, randN) \
  reduction(+:hist_array[:numBins]) default(none)
  {
     // initialize random number engine
     // Note a different choice here from the sequential version.
     // Try different engines and distributions and note the results.
     // see page 25 of the trng.pdf document for list of parallel and
     // sequential engines.
    
    trng::lcg64_shift RNengine1;
    // trng::lcg64 RNengine1;
    // trng::yarn2 RNengine1;

    if (useConstantSeed) {
      seed = (long unsigned int)503895321;     
    } else {  // variable seed based on computer clock time
      seed = (long unsigned int)time(NULL); // enables variation; use for simulations
    }

    RNengine1.seed(seed);

    rank = omp_get_thread_num();
    
    // choose sub-stream no. rank out of nThreads stream
    RNengine1.split(nThreads, rank);
    

    // initialize uniform distribution
    trng::uniform_dist<> uni(min, max);

    // initialize exponential distribution
    // int mean = (max - min)/2;             // just a test for experimenting
    // trng::exponential_dist<> exp(mean);

    for (i = rank; i < N; i+=nThreads)
    {
      randN = uni(RNengine1);
      // printf("r %d ", randN);   // debug

      // for doubles
      // printf("r %f ", randN);   // debug
      // d_index = ((randN - min) / (max - min)) * numBins;
      // index = (int) d_index;
      
      int index = (randN - min) * numBins / (max - min);
      hist_array[index] ++;
    }
  }

  double end = omp_get_wtime(); //ends timer
  double runtime = end - start;

  if (print == 0){
    cout << endl << "RUNTIME: " << runtime;
  }
  else{
    cout << endl;
    for (int k = 0; k < numBins; k++)
    {
      cout << hist_array[k] << "\t";
    }
  }
  cout << endl;
}


