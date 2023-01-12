# Exploring several MPI programs

This repository has several directories with subdirectories inside them. In the terminal, 'cd' into this directory. Then try this command to show what is in each subdirectory:

    ls -R *

There is a set of instructions and description for some of these examples, starting at [this web page](http://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/). There are five examples- it is best to simply follow them in order as shown here. The one example that is not described on the above web page is the heat2D example- for that one use the instructions below in this file.

Remember that there are links in the upper and lower right that take you to the next and previous of these examples on the above web page.
  
For each of them, study the code so that you understand how the parallelism is developed. For each, determine:

1. What patterns are being used?
2. How did those patterns get implemented?
   
Compare the code to what you saw in the patternlets, starting in [chapter 2 of the book you went through on the Raspberry Pis](https://www.learnpdc.org/RaspberryPi-mpi/02ProgramStructure/toctree.html).

## Important note

In the description of each of these examples in the web page above, the file name to examine is given with a prefix like this:

    MPI_examples/

You can ignore this part of the path, as this cloned repository represents that directory.

# monteCarloPi, the Monte Carlo estimation of pi

- folder `monteCarloPi`
  - On the web page for this one it suggests that you can try determining the speedup by making graphs with some suggested problem sizes. If you wish, you can simply try some examples, record them, and determine whether you think it is strongly and/or weakly scalable without doing the full analysis via experimentation and Google spreadsheet. A spreadsheet still might be useful if you are comfortable with that.

  ## Important Updates
  Your version of the code uses trng to generate random numbers. Study the code from your repo, which has a different and improved version of the Toss function from that shown on the above web page. Note that block splitting was chosen for the random numbers assigned to each process. **This means that the number of tosses chosen should be a multiple of the number of processes chosen.** This is because equal division of the blocks split is necessary for the trng library.
  
  Also, in your version, the end time is obtained in a different place than for the code shown in the web page explaining it.

  Also pay close attention to how the code is timed in MPI. The barrier is important to getting the correct time between all processes, because all should start working at the same time. Then, to get the correct finishing time, the last process to finish represents the parallel time, so we reduce to find the maximum time taken by all.

  To run the sequential version of this code on our mscs1 server, do this:

    `cd calcPiSeeq`
    `make`
    `./calcPiSeq 20000000`
  
  Then change the number of tosses by doubling to 40000000, then again to 80000000. You should see the time multiply by approximately 2.

  To run the MPI version of this example on our mscs1 server with a large number of random 'tosses', do this:

    `cd calcPiMPI`
    `make`

    `mpirun -np 2 ./calcPiMPI 1073741824`

  Then try with 4, 8, and 16 processes. What do you observe about the strong scalability?

# trapIntegration, the trapezoidal rule integration using MPI

as given in the Pacheco textbook, section 3.2

- folder: `trapIntegration`
  - For this one, you can experiment with various sizes of number of trapezoids and numbers of precesses. Go to the trapIntegration/mpi_trap folder.

# heat2D, simulating heat propagation on a 2D plate.

This example is worth studying, because it is similar to your homework 2 problem.

- folder: `heat2D`

  - make creates one executable called mpi_heat2D
  - run some examples like this:

  mpirun -np 2 ./mpi_heat2D 

  mpirun -np 3 ./mpi_heat2D 

  mpirun -np 5 ./mpi_heat2D 

  mpirun -np 9 ./mpi_heat2D 

  mpirun -np 2 ./mpi_heat2D -x 1000 -y 1000 -s 2000

  mpirun -np 3 ./mpi_heat2D -x 1000 -y 1000 -s 2000

  mpirun -np 5 ./mpi_heat2D -x 1000 -y 1000 -s 2000

  mpirun -np 9 ./mpi_heat2D -x 1000 -y 1000 -s 2000

Check with your instructor about displaying results of running this code.

# oddEvenSort, an MPI implementation of this sorting algorithm 

as given in the Pacheco textbook, section 3.7.2

- folder: `oddEvenSort`
  - You will need to study this to see how the list to be sorted is generated randomly and what to put into the command line to accomplish this.

# mergeSort, an MPI implementation of this classic sorting algorithm

- folder: `mergeSort`



For the sorting algorithms, note that these can be hard to get really strong scalability. Try several long lengths to see what sort of performance there is. There are things we can try to think about about the general run time. See if you can think about these questions:

1. For odd-even sort, can you determine whether its time does seem to be around O(N)?
2. What about for merge sort - what should its time be for any given N?

