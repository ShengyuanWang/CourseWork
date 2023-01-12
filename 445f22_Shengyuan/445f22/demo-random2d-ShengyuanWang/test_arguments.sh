#!/bin/bash -x

# Runs a series of tests to determine that the getArguments() function
# is working properly. The last 4 tests are correct runnings of the program.
# The others should fail because of incorrect input.


./2d_matrix_init -n
./2d_matrix_init -m
./2d_matrix_init -n -c
./2d_matrix_init -i
./2d_matrix_init -i -c
./2d_matrix_init -i 20 -b
./2d_matrix_init -m 100 -n



# threaded
# ./2d_matrix_init -t
# ./2d_matrix_init -t -p
# /2d_matrix_init -t -n 10

echo "correct runs:"

./2d_matrix_init 
./2d_matrix_init -v
./2d_matrix_init -c
./2d_matrix_init -i 20 -p 0.4
./2d_matrix_init -m 20 -n 20 -i 20 -p 0.4

# ./2d_matrix_init -t 4 -n 50 -m 10 -i 1 -b 1000
# ./2d_matrix_init -t 4 -n 50 -i 10 -i 1 -b 1000 