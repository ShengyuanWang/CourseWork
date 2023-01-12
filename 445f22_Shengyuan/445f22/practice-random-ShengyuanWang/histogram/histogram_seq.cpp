/*
 * Generate pseudo-random numbers using TRNG library and creates
 * an array of "bins" that stores the histogram information
 * Usage: ./histogram_seq -n [numThreads] -i [numBins] -a [min] -b [max]
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <omp.h>

//////////// NOTE  how we need to include the generator and distribution include
////////////       files from trng library
#include <trng/mt19937_64.hpp>
#include <trng/mt19937.hpp>
#include <trng/lcg64_shift.hpp>
#include <trng/normal_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/exponential_dist.hpp>

// separate file for handling command line arguments
#include "./utils/getCommandLineSeq.h"

using namespace std;  // for 'cout' style of printing in C++

int main(int argc, char *argv[])
{
  //initialize parameters and set their default values
  int N = 100000000;
  int numBins = 50;
  // use these to experiment with a distribution of doubles
  // double min = 0.0;
  // double max = 100.0;
  int min = 0;
  int max = 10000;
  int print = 0;
  int useConstantSeed = 0;

///////////////////==========================================
  // command line arguments
  getArguments(argc, argv, &N, &numBins, &min, &max, &print, &useConstantSeed);
  //debug
  // printf("%d values between %d and %d placed into %d bins\n", 
        //  N, min, max, numBins);

  //initialize the array that will hold counts in 'bins'
  int bins [numBins] = {};

  // random numbers start from a seed value
  long unsigned int seed;  // note for trng this is long unsigned

  double start = omp_get_wtime();  // start timing

/////////////////// random engine ==================================
   // initialize random number engine
   // Note there are several choices- see trng documentation:
   // trng.pdf, p. 25
  trng::mt19937_64 RNengine1; // a Mersenne twister one
  // trng::mt19937 RNengine1;
  // trng::lcg64_shift RNengine1;

  // same constant seed will generate the same sequence of random numbers
  // use for testing to varify same sequence regardless of number of threads
  if (useConstantSeed) {
    seed = (long unsigned int)503895321;     
  } else {  // variable seed based on computer clock time
    seed = (long unsigned int)time(NULL); // enables variation; use for simulations
  }

  RNengine1.seed(seed);

///////////////////  distribution ===================================
  // initialize uniform distribution
  trng::uniform_dist<> uni(min, max);   

  // initialize exponential distribution
  // int mean = (max - min)/2;             // just a test for experimenting
  // trng::exponential_dist<> exp(mean);
  ////// end exponential distribution

/////////////////// new random value placed in a 'bin' ==============
// find the correct bin for a new random number generated from the
// distribution.
  for (int i = 0; i < N; i++)
  {
    // uniform:
    int randN = uni(RNengine1);

    // exponential:
    // int randN = exp(RNengine1);

    // debug if you want to see each value randomly generated
    // cout << "r" << randN << " ";

    // next two lines if you want to try double values instead of ints
    // int d_index = (randN - min) * numBins / (max - min);
    // int index = (int) d_index;

    // index for bin where value belongs
    int index = (randN - min) * numBins / (max - min);
    bins[index] ++;
  }

  double end = omp_get_wtime();
	double elapsed_time = end - start;
  
  if (print == 0){
    cout << endl << "RUNTIME: " << elapsed_time;
  }
  else{
    cout << endl;
    for (int k = 0; k < numBins; k++)
    {
      cout << bins[k] << "\t";
    }
  }
  cout << endl;
}

// You could experiment with a distribution of doubles check in
// utils/getCommandLineSeq.c and update to add a function that
// take doubles for range
// the resulting random value needs to be declared as a double.

