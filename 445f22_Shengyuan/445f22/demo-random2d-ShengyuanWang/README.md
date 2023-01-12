# COMP 445 Practice generating random numbers and better command line argument handling

Libby Shoop

This work is based on the version by John Burkardt found here:

 [https://people.sc.fsu.edu/~jburkardt/c_src/ising_2d_simulation/ising_2d_simulation.html](https://people.sc.fsu.edu/~jburkardt/c_src/ising_2d_simulation/ising_2d_simulation.html)

The concept here is to generate a set of random numbers for a 2D matrix. For this code, however, the 2D structure is 'flattened' into a 1D array. This is a somewhat common practice. As a very optional exercise, you could try changing this code for 2D bracket notation for the arrays. It is useful to point out, though, that the location of each element in the underlaying memory in the same in C/C++, because 2D arrays are stored in row major order, and this code mimics that concept.

This type of 2D matrix is designed to then be used in a simulation called an Ising model. We will examine that later in a separate activity or project.

## First make the code and try it

    make
    ./2d_matrix_init -v -c
    ./2d_matrix_init -v -c -t 2

Study the code in the file `2d_matrix_init.cpp`. Note how this version is designed to demonstrate how the random numbers end up getting assigned to the array representing the 2D matrix.

Note that -c also sets a fixed seed so you can tell if the random number generation can be repeatable. However, the resulting 2D matrix is different in the way the numbers are located.


## Creating an output file to visualize results

    gnuplot ising_2d_initial.txt

The above should create a file called ising_2d_initial.png, which you can get to your machine by syncing remote to local in VS Code.

Once you have synced it beck to your machine, you should be able to open it and display it. Match what you see to the code. The display is for the array called `c1` in the main() function of `2d_matrix_init.cpp`.

## Try some different parameters

Try these examples, creating the the .png file for each. To save each result, you could make a copy of it on your laptop before running gnuplot again and syncing remote->local

    ./2d_matrix_init -t 4 -n 50 -m 10 
    gnuplot ising_2d_initial.txt

Sync remote->local in VSCode. Make a copy of ising_2d_initial.png.

    ./2d_matrix_init -t 4 -n 50 -m 10 -p 0.2
    gnuplot ising_2d_initial.txt

Now compare the first one to the second.

## Optional: Testing Command line argument handling

Handling of command line arguments is in the file getCommandLine.cpp. Take a look at it.

Run a series of tests of this like this on the command line:

    bash -x test_arguments.sh

Note some arguments are designed for later use, such as -i for iterations.

Update the test file to cover more cases.