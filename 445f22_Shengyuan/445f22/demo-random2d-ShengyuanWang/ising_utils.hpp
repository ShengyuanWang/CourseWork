// Utility functions for the Ising 2D simulation.
//
// Separated into ising_utils.cpp and ising_utiles.hpp
// by Libby Shoop
// February, 2022
//
void printSimInputs(int m, int n, int iterations, double thresh, long unsigned int seed, double prob[]);
void timestamp ( );

int i4_wrap ( int ival, int ilo, int ihi );
int i4_modp ( int i, int j );
int i4_min ( int i1, int i2 );
int i4_max ( int i1, int i2 );
