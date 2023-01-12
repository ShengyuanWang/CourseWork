# COMP 445 Homework 3: OpenACC with multicore and GPU

## Game of Life example

[See the wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) or do more web searching to see the way this classic cellular automata grid example works.

There are 4 folders with 4 versions of basically the same code:

1. sequential in 1-sequential
2. gcc openMP version in 2-openMP
3. pgcc multicore version in 3-multicore
4. pgcc GPU version in 4-openacc-gpu


Your goal is to use a profiler-driven approach to analyze the original sequential code and create a parallelized version for our NVidia card on mscs1. You will report on the 'hot spots' you find in the code that can be parallelized. Hot spots are points in the code where a great deal of time is being taken by the program. You should use a profiler to find these, then experiment with parallelization and then optimization of your code for multicore and for the GPU. Use the practice labs and the code for them as a guide.

## A sequential version to profile using gprof

Folder: 1-sequential

Study the initial sequential code, `gol_seq.c` along with the main program file, `gol_main.c`and look at how Conway's game of life was coded in this version.

For a sequential version, the gnu profiler program is a better profiler that provides useful information about how much time was spent in each function. This is how you can use it:

    make gol_seq_prof

Note that we use a special compile flag, -pg, for rpofiling, and we don't use any optimization in this version. This version is solely for profiling which functions are taking more time. Run it (be patient, noting the values of number of live cells at points during the 500 iterations):

    ./gol_seq_prof -i 500 -v

Running the code creates file called gmon.out. Use that with the program `gprof` to generate a profile and place it into a file called profile.out, like this:

    gprof ./gol_seq_prof gmon.out > profile.out

Now you can bring this file called `profile.out` back to your local machine by doing 'Sync Remote to Local' on your repo directory.

Observe which functions take the most time by looking at this profile.out file. This gives you an idea of which functions and the loops within them that are good places to start adding pragmas for parallelism. Find the function called `gol`. Which of the following functions in there should you spend time trying to parallelize?

- init_ghosts
- apply_rules
- update_grid

Also note that the init_grid function takes very little time, so we will not worry about parallelizing the loop for that.

In your report, note your choices for functions to concentrate on to parallelize.

### A sequential version from gcc without profiling

This will compile sequential version using gcc without fast optimizations:

    make gol_seq

You can use this as a baseline when you compare the parallel versions to it. Note that it does run quite a bit faster than the profile version- this is because profiling adds extra time to the computation.

## Work on 3 parallel versions

There are 3 folders you will work on:

- 2-openMP
- 3-multicore
- 4-openacc-gpu

You need to add pragma lines from openMP to create an openMP version, then add pragma lines from openACC to create a multicore version and a version that will run kernel functions on the device.

Use what you know from the profiling of the sequential code to improve the two functions that are taking up the most time.

One thing to note about your version of this code for multicore and GPU is that you will consider the grid as a whole, not splitting it into segments as was needed for the MPI version. There are still ghost rows, but they are merely on the top, bottom, and sides of the entire 2D array.

You will report on the improvement in running time from the sequential to the parallel, including an openMP and an openACC multicore version for comparison to the GPU.

There is already a sequential version you can build for comparison:

    make gol_seq

Note in the Makefile how this one gets built without the extra profiling compiler flags and by using pgcc instead of gcc. Run it like this with a number of iterations that we used above:

    ./gol_seq -i 500 -v

Notice the values reported for the number of live cells every 50 iterations. You will want to ensure that your parallel versions also report these same values.

![](./images/exclaim.png) Record the results and times so far for your report.

## Use a profiler-driven Approach: openMP first

Folder: 2-openMP

Note that in this version the threads can be set in main.c with the -t option.

Start by working on an openMP version. First parallelize the function that took the most time in the profiled sequential version. This is in the file called `gol_omp.c`. Make sure that the parallelization is working by ensuring that you get the same results. You should be able to build this version using make:

    make

 For multicore versions of code that work on 2D arrays, it is best to assign multiple rows of the 2D matrix to each thread. Use this notion to decide where the omp pragma should be located in the code.

![](./images/exclaim.png) Be careful to think about what values need to be private, shared, and reduced. Use default(none) to help you get every variable correct.

Try the program similar to the sequential one, but with different numbers of threads:

    ./gol_omp -i 500 -t 1 -v
    ./gol_omp -i 500 -t 2 -v
    ./gol_omp -i 500 -t 4 -v
    ./gol_omp -i 500 -t 8 -v
    ./gol_omp -i 500 -t 16 -v

You should see improvement, but there is still more that can be improved. The point is that you work on one portion of the code at a time and check that your changes still produce correct results.

![](./images/exclaim.png) Record the results and times so far for your report.

Once you have worked on the loop in the function that took the most time in the sequential version, then work on the function that took the second most time. Run the same tests as given above and note the improvement.

![](./images/exclaim.png) Record the results and improved times for your report.

## A pgcc multicore fast version

Folder: 3-multicore

Now let's try the pgcc compiler, targeting the multicore host and using their -fast compiler optimization. Look at the Makefile for how this code is compiled.

Note that in this version the threads can be set in main.c with the -t option.

Do the same as you did for the multicore version: change each function one at a time, using the openACC pragma for loops.

![](./images/exclaim.png) Important Notes:

- In an acc loop pragma for multicore, there is no `default(none)` used.
- In an acc loop pragma, you declare only private variables and reduction variables, and the rest are assumed shared.
- Using the -fast optimization will mean that this version will be faster than the openMP version without such optimization.

![](./images/exclaim.png) As before, record the results and improved times for each of two parallelizations that you add for your report. Do similar timings so you can compare to the openMP version without compiler optimization:

    ./gol_mc -i 500 -t 1 -v
    ./gol_mc -i 500 -t 2 -v
    ./gol_mc -i 500 -t 4 -v
    ./gol_mc -i 500 -t 8 -v
    ./gol_mc -i 500 -t 16 -v

Again, do this for each improvement you make to each of the two functions.

## openACC device version

Folder: 4-openacc-gpu

Now let's work on the openACC version for the GPU device. The file `main.c` has been changed to ignore the -t flag for number of threads. The Makefile has been updated to build an accelerator version with the proper command line flags to the pgcc compiler.

You will once again add the proper pragma lines for openACC to update the loops. As before, start with the loop that should make the most impact and make certain that it is working by comparing the results for the number alive at every 50 iterations. This should give the same results, but the timing may be disappointing:

    ./gol_acc -i 500 -v


## ![](./images/exclaim.png) What's happening?

If the time was not very fast, you need to work on the compilation. In certain cases, the pgcc compiler will need you to declare that each nested loop is truly `independent`. In this case, you need to signify that the inner loop does not have data dependencies on the arrays grid and newGrid.

First go back and observe the compiler output by cleaning and re-compiling:

    make clean
    make

Look carefully to see if it contains results like this:

```sh
Complex loop carried dependence of grid->,newGrid-> prevents parallelization
```

To sort this out, you need to add this to the inner loop:

    #pragma acc loop independent

You will find that this will be true for the second function that you need to parallelize for the GPU device.

Record the time after you have each nested loop in each of the two functions parallelized properly. A correct result for this:

    ./gol_acc -i 500 -v

should complete in approximately 4.7 seconds.

Another good check is to profile this code and verify that most of the time is on the GPU cores in the functions that you updated. Run the profiler like this:

    nvprof --cpu-profiling off --openacc-profiling on ./gol_acc -i 500

## Now its time to study scalability

Now that we have working versions of the various parallel codes, we can eliminate the printing of number alive at every 50 iterations. This will let us try larger problem sizes.

Problem size is in 2 dimensions: 
1. iterations of the game board
2. number of cells in the grid

For this problem, we have written the code to have a square grid. The default initial size is 2048x2048, or 4194304 cells. We can approximately 'double' the grid size by doubling this number of cells and taking its square root. This works out to be 2896.

    ./gol_acc -i 500
    ./gol_acc -i 500 -n 2896
    ./gol_acc -i 500 -n 4096

What do you notice about the timing as we double the problem size each time? Is this version scalable?

Now go back to the 3-multicore folder and try some similar timings:

    ./gol_mc -i 500
    ./gol_mc -i 500 -t 8
    ./gol_mc -i 500 -n 2896 -t 8
    ./gol_mc -i 500 -n 4096 -t 8

And also try the gcc openMP version in 2-openMP, which is less optimized:

    ./gol_omp -i 500 -t 8
    ./gol_omp -i 500 -n 2896 -t 8
    ./gol_omp -i 500 -n 4096 -t 8

It will take some time, but also record sequential times for these cases.

![](./images/exclaim.png) These versions show some interesting results. The GPU card version likely is not faster than your optimized multicore version. In your report, consider why this might be. As a hint, consider how much actual computation is being done in these loops compared to the laplace method of updating the temperature values in each cell of the simulated plate done in the openACC activity.

-------------



## Write a report

Make a useful, well-written report that describes your code optimizations and the results. Below are some items you should include. You can chose the order in which you present the following ideas in your report, using sections with headers.

For goodness sake, own your work by putting your name on your report!!!

Devise a title for your report.

Explain briefly what problem you are analyzing and optimizing- orient the reader of your report to the problem.

Explain what you discovered when profiling the sequential version that led you to decide what part of the code to parallelize.

Explain the code updates that you made.

Explain what the 'problem size' is and how you changed it and your observations about performance as it varied.

Write about the experiments you tried and what improvements worked better. Write an explanation of why the GPU version performed the way it did.

The concept of speedup is different for code on accelerators. Using many smaller, slower cores speeds up code in a different way so that traditional speedup does not apply. Use other methods to show the change in speed, such as bar charts that compare each of the four versions for varying problem sizes. (Recall that the practice-openacc lab activity provided a simple spreadsheet to create bar charts.)

Ultimately, what did you learn by parallelizing this example with these different compilers and optimizations and the difference between the fast CPU and slower cores on the GPU? 



