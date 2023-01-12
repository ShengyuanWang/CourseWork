# OpenMP Homework: Counting sort

You will examine a well-known algorithm for sorting called counting sort. It is used most often historically when the range of values, or keys, is relatively small compared to a much larger set of items within that range are to be sorted. Thus, the final sorted list will have many duplicates. This seems to be an algorithm that is used on company coding exams-- look at [this example of a site designed to let you practice writing this algorithm for such an exam](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/).

You are given a sequential version of this code in a file called **countSort_seq.c.** This particular version is assuming that we will be counting characters in a string. The range of printable ascii characters is fairly small, and the input strings to be sorted will be quite large.

In this document you have a guide for the way we should work on examining the sequential code and parallelizing it, by looking at what parts of the code contribute to the overall time and working on each one incrementally to parallelize it, trying it out before moving on the the next part.

**Note:** You will be creating a report for this project. Please note what is expected at the end of this readme file.

Before you start on parallelizing, let's also look at a sophisticated example of how we gather command line arguments from the terminal. This involves a way of gathering what the user types and checking whether required arguments were given.

## Learn how to handle command line arguments

As a prelude, let's examine a program available in linux that has a lot of command line arguments: ls.  Try this at the command line in the terminal on mscs1, from the directory where this code is:

    man ls
    (hit space bar to scroll up a page; hit the q key to quit)

Notice there are a great many different 'flags' that we can set at the command line. Each is listed with an explanation. We can see a few of them in action:

    ls -a

In this above -a is called a flag. We can use several flags at once in two ways:

    ls -ltc
    ls -l -t -c

Also try it with the C compiler, gcc, like this, to see some errors:

    gcc -o
    gcc -o countSort_seq.c

This happened because the -o flag also needs a file name for the executable file. in the second case it wrongly interpreted getopt.c to be the output, but then concluded it had no input C code file.

----
### How we will write solid C code for arguments

There is a folder called utils with the following files in it:

    getCommandLine.c
    getCommandLine.h

This shows the way we handle command line entries in well-designed, solid C programs. Note this is based on the example in the HW 0 practice you hopefully have already completed.

If you want some details about handling command line arguments start by looking through [the first four chapters of this complete explanation of the getopt function and how to use it](https://azrael.digipen.edu/~mmead/www/Courses/CS180/getopt.html). Read carefully through at least the section on unknown and missing option arguments. 

----

The examples in the file utils/getCommandLine.c given to you here shows how to use getopt(). *Look carefully at this code and the comments that have been added to it.*

Note that the call to `getopt` inside the while loop has a string as the third argument:

    while ((c = getopt (argc, argv, "n:vh")) != -1) {

In that string, note the use of the : after n. This indicates that this program expects a string after the -n flag. *This will be useful to you in many of the parallel programs, as we often want to state a problem size and number of threads to use when we run it.*

Now let's work on a sequential and parallel character sorting algorithm!

## Observe the Makefile

Have a look at the Makefile. Note the following:

- We will compile using the gnu C++ compiler, g++. this is because you will need this compiler when you create the openMP version and use the trng library for random numbers. 

- As for the activity regarding random numbers, there are special compiler flags needed to bring in the trng library.

- You will eventually make two different executables once you have created your parallel version in a new file.

## Test the sequential version

In the main() function, note the printf statements that can be generated with the -v flag.

Then make the sequential version 

    make

and test it with a small string, like this:

    ./countSort_seq -n 20 -v

Run it again just as above and note that the generated input string is the same. This is because we used a fixed 'seed' for the random number generation. Find that in the function called generateRandomString(). Next try this:

    ./countSort_seq -n 40 -v

Observe that the first 20 characters remain the same as the previous input string.

Make sure that it appears that the string is being sorted according to ascii character values. Refer to (this ascii table)[https://www.asciitable.com/] to see the values for each ascii character.

## Try large problem sizes

Now avoid the debug print statements and try larger values of the length of the input string, like this:

    ./countSort_seq -n 8388608

This creates a string that is 2^20 characters.

# Profiling code
Run it several times using various powers of 2  from 2^20 to 2^30 (1073741824) There is a list of the powers of 2 in two files (one excel sheet, one pdf) in the Resources section of moodle. Observe the time it takes for each section of the code that is doing three main tasks:

1. Generating the input string
2. Counting the number of occurrences of each character in the input string
3. Generating the output sorted string "in place", reusing the memory used for the input string.

This type of timing is called **profiling**. We want to see what takes time and how we can improve it using parallelism.

## Keep records for comparison
Be sure to record your observations as part of the report that is required for this homework (see below).

# Make your own OpenMP version

Use a copy of the sequential version as your starting point. 

Option 1: In terminal:

    cp countSort_seq.c countSort_omp.c

And then synchronize it back to your editor. (sync remote -> local)

Option 2: create a copy with your editor and ensure that it is syncing to the server by using sync local->remote on the repo folder.

## Fix the Makefile
Note the portions of the makefile that pertain to the compiling of countSort_omp.c. Also note this on line 6:

    TARGETS = countSort_seq #countSort_omp

Remove the #  comment so that you will be able to build the countSort_omp.c file.

Now you should be able to follow the steps below to incrementally build up your OpenMP version.

## Take in the number of threads on the command line

Declare an integer variable nThreads. Set it to 1 as the default.

Change the call to *getArgumentsSeq()* to *getArguments()* so that you can take -t option for a number of threads, nThreads. Look in getCommandLine.c for how the function call should include a pointer to nThreads. If -t is not supplied, nThreads should default to 1.

Be sure that you use the openMP library function to set the number of threads once you have the number of threads the user wants or the default value. Refer to HW0 as an example. Print out this value if verbose output is chosen.

## Work one loop at a time

There are 3 functions in the sequential code that have the possibility of using the **parallel for loop** implementation strategy for data parallelism. Work on each fix needed, testing as you go.

### 1. Input data creation

The *generateRandomString()* function is an artificial function, in that in real use you might be reading in a stream of characters from some data source. Reading in data in parallel is an open topic of research in parallel computing, so we will stick to artificially generating random values for now.

#### Random values

The best option for generating random characters in parallel will be to use the trng library that was part of a practice activity. Refer to code examples and the Makefile for compiling and using this library.

Note in the sequential version, the rand_r function was used, but this is **NOT** thread safe. Also note that to get a value within a range, we did this:

    ((rand_r(&seed) % (ASCII_END-ASCII_START)) + ASCII_START)

With trng it is much simpler to specify a range of values by using a uniform distribution, something like this:

    trng::uniform_dist<> uni(ASCII_START, ASCII_END);

Also note that a seed is a long unsigned int with trng, a fixed one could be larger, like this for a dynamically changing one:

    long unsigned int seed = (long unsigned int)time(NULL);

Or like this for a fixed one:

    long unsigned int seed = 1403297347956120;

Make sure you can compile and run for this change before going on to the next section. Test for accuracy with short strings.

**Top Tips**:

Start with a fixed seed so the same string gets created each time. Try a small problem size and use one or two threads, printing the input and output. Make sure your multi-threaded version  works the same as your single-threaded version. Do this as you make each of the updates needed.

Try a larger string without -v, with 1 thread and then 2 threads and 4. be certain that the time for this first portion of the code decreases as you use more threads.

### 2. Generating Counts

Now focus on the function called *countEachLetter*. There is a race condition here on the entire array called counts. Our version of gcc installed on mscs1 allows a new feature of gcc: reduction using elements in entire arrays. Try to look up how this is done in OpenMP. There aren't a lot of references to this online yet, but hopefully you can find some. Consult your instructor if you have problems with this (you don't want to have to do the reduction yourself).

### 3. Creating the final sorted output string

This one is harder to parallelize directly. The problem is that each time through the outer loop in the *createdSortedStr* function, a different amount of work is done in the inner loop. For a parallel version, you are going to have to remove the dependence on the variable r, which in the inner loop is designed to go sequentially through the output array. You want each thread to know what small portion of output it can fill independently of the other threads.

#### Prefix sum to the rescue

To parallelize the outer loop that creates the output, you will need to ensure that each thread is working on particular sections of the final output. Creating a *prefix sum* array from the counts array is a really good way to do this. You will need to read about this technique and decide how you can then use the prefix sum array instead of counts in the inner loop to ensure that each thread can place the correct number of characters. Here we will be adding new code to ensure that we can complete this task in parallel. When you complete it, it will be a good idea to see if it is worth it.

Since counts is small and the prefix sum array that we make from it is small, we can use a sequential version of prefix sum, such as one like [this described in a Geeks for Geeks post](https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/). As an aside, note some of the other applications of this technique that this post describes.

# Sequential vs. 1 thread

You may find that your OpenMP version using 1 thread on a particular problem size takes quite a bit different time than the original sequential version given to you. This is a mystery that is hard to unravel, because we don't know just how the compiler is creating the threaded version of the code and optimizing it in some way. It is most likely due to these factors:

- The difference in speed of the thread-safe trng number generation library is significant. You can observe this by checking what the time is for generating the input string.
- There were other changes that you made to the threaded version to use the prefix sum technique. You can likewise observe this by looking at the time for creating the sorted string.

*If you find this is the case, use the threaded OpenMP version with one thread as your sequential case when computing and reporting speedup and efficiency.*

# Testing

The original prints of time are necessary so that you can observe how each portion of the code contributes to the overall time. For a report on scalability *of simply sorting the string*, you will need to update the printing and run experiments using various input sizes, which you print on one line with the number of threads and overall time **to sort the string**. 

-----

![](img/small_warning.png) It is useful to use parallelism to generate these large strings, since it takes time. It may be equally useful to know what the scalability of just the sorting of those strings is. This is because in the wild the sorting may be done on real data strings that are not randomly generated. Therefore, you should ultimately conduct your strong and weak scalability analysis on the *sorting portion only.* For your report, update the code as described below by changing the timing to eliminate the generation of the input string.

-----

## Some things to consider

![](img/small_warning.png) Be sure that you do not use a fixed seed for generating random characters when you run your final experiments. You want variety in your input data when analyzing your parallel implementation. This detail is something that is also useful to state in your report.

It will be best to be thorough and try many problem sizes and for speedup and consideration of strong scalability. Use this sequence of the number of threads: 1,2,4,6,8,12,16. However, you may run into some anomalies with 6 and 12 threads. If you do just report them. Note that for a shared memory system like this, there are rarely any gains past 16 threads, and often well before that (it varies with problem size).

For the parallel version of reduction on the counts array, it is interesting to realize that the counts array is duplicated on each thread. This shouldn't cause memory issues, however.

There is a limit to how many random characters you can generate. The sequential rand_r() and the trng parallel approach both have this issue. From testing, it appears that this is the maximum value you can try:

    ./countSort_seq -n 2147483616
    ./countSort_omp -n 2147483616 -t 16

This value is very close to 2^31, so you can use it as that value for n.

For the weak scalability case, you will need to devise some cases where you can proportionally increase the problem size along with the number of threads and still be able to generate results. The top value after doubling the number of characters for each case will have to be less than 2147483616. This is for line 3 of these weak scalability graphs- your analysis should go as high as you can go in problem size.

You should run some initial experiments with the separate timings each of the 3 portions of the code in place to observe which of those portions scale better than others. Report this as an observation about the code.

You should consider this when writing your final report: you will definitely want to report about the parts that count and create the sorted string separate from the original generation of the input, which is artificial and may change when applied to a real situation. You should make clear what was included in the time of any of your resulting graphs. At a minimum, include scalability for sorting. If you wish to also examine scalability of creating the string separately, or ultimately both as a separate optional analysis, you can do that.

For reporting on how your code performs on various input sizes and numbers of threads, you will want to change how your code prints results so that you can create data that can easily be copied into a spreadsheet. Recall the trapezoid practice activity: use bash shell scripts like that one did.


![](img/small_warning.png) You will need to update the bash shell scripts that will run each test that you devise several times. The problem sizes that you want to run for this sorting problem are likely different than those used for the trapezoid example. Also, you will likely want to simply use the omp version with 1 thread rather than the sequential version for the strong scalability analysis.

You are free to write scripts using python also, if you would prefer that. You should be able to import the results of running those scripts into your spreadsheet and then work to get the speedup, efficiency, and weak scaling plots.


# Report your results
You will write a report for every homework assignment and your project that will explain what you did, how you ran experiments to analyze its performance, and what your results were.

Your report must be placed in the Google Drive Folder that I shared with you for your course work.

Also make certain that your code gets pushed up to github for review.

## Report Criteria

-   Your name
-   Well-written prose explaining:
      1.  How you profiled the sequential version of the code and what you discovered.
      2.  Updates to the code sections to enable parallelism.
      3.  What parallel patterns you used in the code updates.
      4.  How to compile the code.
      5.  How to run the versions of the program.

-   Well-written prose explaining the methods you used to obtain your data:      
      1.  What scripts you used to generate your data.
      2. Description of your methodology: how many times you ran your experiments and what conditions you used (i.e. cases you tried).

-   Well-written prose containing your findings regarding the speedup of the parallel version, including:   
      1.  Clear explanation of test cases you tried,
      2.  Graphs depicting strong scalability via speedup and efficiency and weak scalability, and
      3.  Explanation of results.


Note that the report should have sections with headers so that each part of what you are reporting is obvious to the reader (they should not simply be named exactly like the bullet points above- think about how you want to organize your work).

When writing the explanation of your results, address at what conditions the program exhibits strong scalability and weak scalability, backing your assertions up with references to your experimental results.
