#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "./utils/ParGetCommandLine.h"

/* Demo program : computes trapezoidal approximation of an integral*/
/*
 *  The integral from 0 to pi of the function sine(x) should be equal to 2.0
 *  This program computes that using the 'trapezoidal rule' by summing rectangles.
 */

const double pi = 3.141592653589793238462643383079;

double f(double x);   // declaration of th function f, defined below

int main(int argc, char** argv) {
  /* Variables */
  double a = 0.0, b = pi;  /* limits of integration */;
  unsigned long n = 1048576; /* number of subdivisions default = 2^20 */

  double integral; /* accumulates answer */
  int numThreads = 1;  /* number of threads to use */
  // for this program we will print human readable results by default
  int verbose = 1;        
  int experimentPrint =0;

  /* parse command-line args */
  getArguments(argc, argv,
                  &n, &verbose, &experimentPrint, 
                  &numThreads);
  
  // remove the default verbose printing if want experiment printing
  if (experimentPrint) {
    verbose = 0;
  }

  double h = (b - a) / n; /* width of subdivision */

// illustrates that you can check whether the compiler is capable of openMP
#ifdef _OPENMP
  if (verbose){
    printf("OMP defined, numThreads = %d\n", numThreads);
  }
  omp_set_num_threads(numThreads);
#else
  printf("OMP not defined\n");
#endif


  integral = (f(a) + f(b))/2.0;
  int i;

  // for timimg
  double start = omp_get_wtime();

// compute each rectangle, adding area to the accumulator
///////////////// 
// We've added the proper additions to the openmp pragma here for correct output.
// Study the pragma line.
////////////////
#pragma omp parallel for default(none) \
private(i) shared(n, a, h) reduction(+:integral)
  for(i = 1; i < n; i++) {
    integral += f(a+i*h);
  }

  integral = integral * h;

  // Measuring the elapsed time
  double end = omp_get_wtime();
  // Time calculation (in seconds)
  double elapsed_time = end - start;

  //output for verbose mode
  if (verbose) {
    printf("With %ld trapezoids, our esimate of the integral from %lf to %lf is %lf\n", n, a, b, integral);
    printf("Time: %lf seconds\n", elapsed_time);
  }
  
  //output for sending to a spreadsheet using bash scripts: just the time
  // followed by a tab
  if  (experimentPrint) {
    printf("%lf\t",elapsed_time);
  }
}

/*
 *  Function that simply computes the sine of x.
 */
double f(double x) {
  return sin(x);
}
