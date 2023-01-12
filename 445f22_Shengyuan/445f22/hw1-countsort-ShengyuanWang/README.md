# OpenMP Homework: Counting sort

You will examine a well-known algorithm for sorting called counting sort. It is used most often historically when the range of values, or keys, is relatively small compared to a much larger set of items within that range are to be sorted. Thus, the final sorted list will have many duplicates. This seems to be an algorithm that is used on company coding exams-- look at [this example of a site designed to let you practice writing this algorithm for such an exam](https://www.hackerearth.com/practice/algorithms/sorting/counting-sort/tutorial/).

You are given a sequential version of this code in a file called **countSort_seq.c.** This particular version is assuming that we will be counting characters in a string. The range of printable ascii characters is fairly small, and the input strings to be sorted will be quite large.

In this document you have a guide for the way we should work on examining the sequential code and parallelizing it, by looking at what parts of the code contribute to the overall time and working on each one incrementally to parallelize it, trying it out before moving on the the next part.

**Note:** You will be creating a report for this project. Please note what is expected at the end of this readme file.

Before you start on parallelizing, let's also look at a sophisticated example of how we gather command line arguments from the terminal. This involves a way of gathering what the user types and checking whether required arguments were given.

## Learn how to handle command line arguments

First examine the code in the file getopt.c. This shows the way we handle command line entries in well-designed, solid C programs. Please start by looking through [the first four chapters of this complete explanation of the getopt function and how to use it](https://azrael.digipen.edu/~mmead/www/Courses/CS180/getopt.html). Read carefully through at least the section on unknown and missing option arguments. You could first do the following suggestions to get the idea of what we mean by command line arguments and what types of arguments there are.

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
    gcc -o getopt.c

This happened because the -o flag also needs a file name for the executable file. in the second case it wrongly interpreted getopt.c to be the output, but then concluded it had no input C code file.

The example in the file getopt.c given to you here shows how to create a portion of your main program, typically right at the top, that enables you to set up flags and required values for your programs. *Look carefully at this code and the comments that have been added to it.*

Note that the call to `getopt` inside the while loop has a string as the third argument:

    while ((c = getopt(argc, argv, "df:mps:")) != -1)

In that string, note the use of the : after f and s. This indicates that this program expects a string after the -f flag or the -s flag. *This will be useful to you in many of the parallel programs, as we often want to state a problem size and number of threads to use when we run it.*

Create the getopt program using make:

    make getopt

Not in the usage of this program, the -f option is required. Watch what you get when you do not include it, and match this up to the code in getopt.c:

    ./getopt

Now make sure that you include that:

    ./getopt -f foo

Oops, in this case it still needs another last input string:

    ./getopt -f foo bar

Now experiment yourself with the other  possible options by looking carefully at the code and reviewing the [detailed description](https://azrael.digipen.edu/~mmead/www/Courses/CS180/getopt.html) that was mentioned above. Try putting the flags in different order. This example should help you with changes you will make to the parallel version of your code. Note in it and the sequential version, we will us -n as a flag that takes a value representing the 'problem size', which in this case is the number of elements in a string to be randomly created, then sorted.

Now let's work on a sequential and parallel character sorting algorithm!

## Test the sequential version

In the main() function, uncomment the printf statements with the comment line preceding them:

```
// debug
```

Then make the sequential version 

    make countSort_seq

and test it with a small string, like this:

    ./countSort_seq -n 40

Make sure that it appears that the string is being sorted according to ascii character values. Refer to (this ascii table)[https://www.asciitable.com/] to see the values for each ascii character.

## Try large problem sizes

Now comment out the debug print statements again and try larger values of the length of the input string, like this:

    ./countSort_seq -n 8388608

Use various powers of 2 (there is a list of them in two files (one excel sheet, one pdf) in the Resources section of moodle). Observe the time it takes for each section of the code that is doing three main tasks:

1. Generating the input string
2. Counting the number of occurrences of each character in the input string
3. Generating the output sorted string "in place", reusing the memory used for the input string.

This type of timing is called profiling. We want to see what takes time and how we can improve it using parallelism.

Be sure to record your observations as part of the report that is required for this homework (see below).

# Make your own OpenMP version

Use a copy of the sequential version as your starting point. 

Option 1: In terminal:

    cp countSort_seq.c countSort_omp.c

And then synchronize it back to your editor. (sync remote -> local)

Option 2: create a copy with your editor and ensure that it is syncing on save to the server.

## fix the Makefile
Note the portions of the makefile that pertain to the compiling of countSort_omp.c. Also note this on line 7:

    all: getopt countSort_seq #countSort_omp

Remove the #  comment so that you will be able to build the countSort_omp.c file.

Now you should be able to follow the steps below to incrementally build up your OpenMP version.

## Take in the number of threads on the command line

Change *getArguments()* to take -t option for a number of threads, nThreads. If -t is not supplied, nThreads should default to 1.

## Use OpenMP timing functions

The OpenMP function *omp_get_wtime()* is designed to work with code where threads are forked. The standard C functions that are used in the provided function *c_get_wtime()* do not create correct times for threaded code.

## Work one loop at a time

There are 3 functions in the sequential code that have the possibility of using the **parallel for loop** implementation strategy for data parallelism. Work on each fix needed, testing as you go.

### 1. Input data creation

The *generateRandomString()* function is an artificial function, in that in real use you might be reading in a stream of characters from some data source. Reading in data in parallel is an open topic of research in parallel computing, so we will stick to artificially generating random values for now.

#### Random values

You have some options for generating random characters in parallel. The best will be to use the trng library that was part of a practice activity. Refer to code examples and the Makefile for compiling using this library.

Another way of speeding up the loop in the *generateRandomString()* function is a bit trickier. You could stick with rand_r, but ensure that each thread gets its own unique seed (using time() alone will not be adequate). Create a function *seedThreads* that takes in an array of unsigned ints called seeds whose length is equal to the number of threads being used. This function should then populate the array of seeds with different values. Though the work this function does is small, each thread can create its own seed in parallel.

Make sure you can compile and run for these updates for this change before going on to the next section. Try a small problem size and use one or two threads, printing the input and output to be sure your threaded version still works. Do this as you make each of the updates needed.

### 2. Generating Counts

Now focus on the function called *countEachLetter*. There is a race condition here on the entire array called counts. Our version of gcc installed on mscs1 allows a new feature of gcc: reduction using elements in entire arrays. Try to look up how this is done in OpenMP. There aren't a lot of references to this online yet, but hopefully you can find some. Consult your instructor if you have problems with this (you don't want to have to do the reduction yourself).

### 3. Creating the final sorted output string

This one is harder to parallelize directly. The problem is that each time through the outer loop in the *createdSortedStr* function, a different amount of work is done in the inner loop. For a parallel version, you are going to have to remove the dependence on the variable r, which in the inner loop is designed to go sequentially through the output array. You want each thread to know what small portion of output it can fill independently of the other threads.

#### Prefix sum to the rescue

To parallelize the outer loop that creates the output, you will need to ensure that each thread is working on particular sections of the final output. Creating a *prefix sum* array from the counts array is a really good way to do this. You will need to read about this technique and decide how you can then use the prefix sum array instead of counts in the inner loop to ensure that each thread can place the correct number of characters. Here we will be adding new code to ensure that we can complete this task in parallel. When you complete it, it will be a good idea to see if it is worth it.

Since counts is small and the prefix sum array that we make from it is small, we can use sequential version of prefix sum, such as one like [this described in a Geeks for Geeks post](https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/). As an aside, note some of the other applications of this technique that this post describes.

# Sequential vs. 1 thread

You may find that your OpenMP version using 1 thread is faster than the original sequential version given to you. This is a mystery that is hard to unravel, because we don't know just how the compiler is creating the threaded version of the code and optimizing it in some way. It might also have something to do with the differences in how the time is being computed (pure speculation at this point).

*If you find this is the case, use the threaded OpenMP version with one thread as your sequential case when computing and reporting speedup and efficiency.*

# Testing

The original prints of time are necessary so that you can observe how each portion of the contributes to the overall time. You will need to update these and run experiments using various input sizes, which you print on one line with the number of threads and overall time.

## Some things to consider

It will be best to be thorough and try many problem sizes and for speedup and consideration of strong scalability use this sequence of the number of threads: 1,2,4,6,8,10,12,14,16. However, you will run into some issues:

For the parallel version of reduction on the counts array, it is important to realize that the counts array is duplicated on each thread. As you increase the number of threads, you will eventually run out of memory for larger sizes of the input string. For example, this case caused a 'Segmentation fault' in my OpenMP version:

    ./countSort_omp -n 1073741824 -t 16

You will find that you are fairly limited at what you can do with 16 and 14 threads. This is an interesting limitation of this approach that you will want to think about when devising your test cases and how you want to report your results. **It will be better to try larger problem sizes that take more time even if in some cases you have a limit of how many threads you can use.** Not all of the curves for each problem size in your charts need to use the same numbers of threads- some can stop shorter than others, and in this case should.

For the weak scalability case, you will need to devise some cases where you can proportionally increase the problem size along with the number of threads and still have enough memory to generate the results.

You should run some experiments with each of the 3 timings in place to observe which portions of the code scale better than others. You should consider this when writing your final report: you will probably want to report about the parts that count and create the sorted string separate from the original generation of the input, which is artificial and may change when applied to a real situation.

For reporting on how your code performs on various input sizes and numbers of threads, you will want to change how your code prints results so that you can create data that can easily be copied into a spreadsheet.

You will need to create bash shell scripts that will run each test that you devise several times (using ones from practice activities as your guide). You are free to write scripts using python also, if you would prefer that. You should be able to import the results of running those scripts into your spreadsheet and then work to get the speedup, efficiency, and weak scaling plots.


# Report your results
You will write a report for every homework assignment and your project that will explain what you did, how you ran experiments to analyze its performance, and what your results were.

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
      2.  Graphs depicting speedup and efficiency and weak scalability, and
      3.  Explanation of results.


Note that the report should have sections with headers so that each part of what you are reporting is obvious to the reader (they should not simply be named exactly like the bullet points above- think about how you want to organize your work).

When writing the explanation of your results, address at what conditions the program exhibits strong scalability and weak scalability, backing your assertions up with references to your experimental results.
