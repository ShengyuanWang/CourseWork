# OpenACC : A new standard and compiler

The software libraries and compilers that we have used so far, OpenMP and MPI, are at their core **standards**. These standards are written to dictate what functionality the compiler and the associated libraries should provide. We have used gcc and g++ for compiling C and C++ code. These compilers conform to the OpenMP standard. The OpenMP standard has several versions, and each successive version of gcc and g++ adds most of the features of the new version of the standard. There are other compilers for C and C++ that also follow the OpenMP standard. MPI libraries and compilers and associated programs like mpirun are also defined in a standard. There are two primary versions of MPI that we can install: MPICH and OpenMPI. You have actually used one of each, as we have OpenMPI on the Raspberry Pis and MPICH on the mscs1 server.

These two standards, OpenMP for shared memory CPU computing, and MPI for distributed networked systems or single machines, each work with traditional multicore CPU computers. As we have seen, they can give us reasonable scalability to perform better and work on larger problems.

Yet many computational problems need much larger scalability. The field of supercomputing exists to develop hardware and software to handle these scalability needs. One aspect of providing more scalability is to turn to special devices that can be installed along with a multicore CPU. Graphics processing units, or GPUs, are such a type of accelerator. Today GPUS come in all sorts of sizes, from small ones for mobile devices and laptops to large ones like this that are separate cards containing thousands of small cores that are slower than a typical CPU chip.

Because GPUs are special hardware with thousands of cores, using them for parallelism is often called manycore computing. They were first designed to off load graphics computations to speed up response time for applications such as games or visualizations. Now they are used for speeding up general computations as well. 

To do this, we will be using a specific **new compiler called pgcc**, based on a different standard called OpenACC. The ACC part of OpenACC stands for accelerator. Separate cards such as GPUs are often referred to as accelerators when they are used for accelerating  certain types of computations. The pgcc compiler is one a several compilers written originally by the Portland Group (PGI), which is now owned by NVIDIA, the major GPU manufacturer and inventor of the CUDA language compilers for running code on GPUS.


## A new compiler

The `pgcc` compiler will process code with pragma lines similar to those of OpenMP into different versions of code:

1. regular sequential version (when no pragmas present)
2. shared memory version using OpenMP  (backwards compatible with OpenMP)
2. multicore version for the CPU    (new pragmas defined in OpenACC standard)
3. manycore version for the GPU     (additional pragmas for accelerator computation)

As you will see, the OpenACC standard and the PGI compilers are designed to enable software developers the ability to write one program, or begin with one sequential program or OpenMP program, and easily transform it to a program that can run a larger sized problem faster by using thousands of threads on a GPU device.

----

# Vector Addition as a simple example

The example we will start with is the simplest linear algebra example: adding two vectors together. We will examine several code versions of this that will demonstrate how the new pgcc openACC compiler can create the different versions of code mentioned above:

| Folder        | Description         |
|---------------|---------------------|
| 1-sequential  | 3 sequential versions, compiled with gcc and pgcc |
| 2-openMP      | 2 openMP pragma versions, compiled with gcc and pgcc |
| 3-openacc-multicore   | 1 openACC pragma version using just the multicore CPU, compiled with pgcc|
| 4-openacc-gpu | 1 openACC pragma version for the GPU device, compiled with pgcc |
| | |

## 1-sequential

folder: vector-add/1-sequential

Examine the Makefile carefully for the sequential version. Note carefully how we can use both different compilers and send command line flags to those compilers to create different programs that have different executable instructions in them. The compiler arguments are supposed to add optimizations to the code.

- for gcc, -O3 (that's a capital O letter, not a zero) is the default compiler optimization level.

- for gcc, -Ofast adds additional optimizations.

- for pgcc, -fast also adds similar additional optimizations. We can see what the optimizations are during compile by adding the -Minfo=opt

On mscs1, go into the directory for the sequential version and make it. Look carefully how the three versions that are made get compiled with different arguments to the compiler. In the case of pgcc, it prints out additional information about what optimizations the compiler performed.

Look at [the Wikipedia page for loop unrolling](https://en.wikipedia.org/wiki/Loop_unrolling) to see a discussion of this technique. Notice what the article said about the size of the executable program. Try this command:

    ls -l

The number just before the data in the listing is the size of the file. What do you notice?

For simple code like this example, these types of optimizations on a sequential version of code may be all we need to create as fast a version as possible.

The pgcc compiler can tell you exactly what optimizations it is adding to the code if you execute the compiler like this on the command line:

    pgcc -help -fast

Run a small example to see that it is adding two arrays, one called x that is initialized with 1.0 in each element, and one called y that is initialized with 2.0 in each element, and putting the addition of those elements back into the y array.

    ./vectorAdd_gcc_O3 -n 20
    ./vectorAdd_gcc -n 20
    ./vectorAdd_pgcc -n 20

The you can try with a default larger array:

    ./vectorAdd_pgcc

Try the other versions also to see that they all run correctly.

The point of this version is to notice these concepts:

- The same code can be compiled with 2 different compilers, creating different executables.

- The executables created have different size.

- Different arguments to the compilers cause different types of optimizations to be included. Since the optimizations may or may not help improve the speed of code, the compiler writers give us the option of using them.

## 2-openMP

folder: vector-add/2-openMP

This version shows the addition of OpenMP pragmas to the code that does the vector addition. Study the code and the Makefile. Note that 4 versions of the code are made. You can choose -n 20 as with the sequential version, and along with it you can now choose -t with a value for the number of threads to verify that the loop is being parallelized.

What you will notice when using the default value for the array size, which is just about as large as we can get using an int to designate its size, parallelizing the loop by using -t has some unexpected performance. Try some cases to see what you observe.

This version illustrates the following:

- We can again use more than one compiler, each with different optimization levels, on code with openMP pragmas.

- For certain simple array calculations, using optimization may or may not help improve performance.

- For certain simple array calculations, using additional threads may not help improve performance.

- In general when considering performance improvement, it takes some care in the use of our compilers. Since most useful programs will have far more code than this, we will need to examine its loops separately when analyzing it and adding improvements.

# 3-openacc-multicore
folder: vector-add/3-openacc-multicore

The openACC compiler called pgcc introduces a set of pragmas for use on the host multicore CPU that are similar to those of openMP, but with slightly different syntax.

This version of the code has these changes:

- The number of threads is set in a different way.
- The pragma for the loop is different than openMP.
- There are slightly different print statements to match the program.
- The Makefile uses different flags with the pgcc compiler. Note this line:
    
    MOPTS = -mp -ta=multicore -Minfo=opt

The -mp and -ta=multicore are indicating that the executable will be built for a multicore CPU (not the GPU).

This version is here to show that you can simply use pgcc pragma directives on loops to have the compiler use threads and execute the loop in parallel on the CPU. 

-----

### OpenACC sidebar

Here is what NVIDIA, one of the  primary developers of the OpenACC standard, have said about this type of compilation using pgcc:

#### Why Use –⁠ta=multicore?
    " This gets to the basic reason why OpenACC was created ... Our goal is to have a single programming model that will allow you to write a single program that runs with high performance in parallel across a wide range of target systems. Until now we have been developing and delivering OpenACC targeting NVIDIA Tesla and AMD Radeon GPUs, but the performance-portability story depends on being able to demonstrate the same program running with high performance in parallel on non-GPU targets, and in particular on a multicore host CPU. So, the first reason to use OpenACC with –⁠ta=multicore is if you have an application that you want to use on systems with GPUs, and on other systems without GPUs but with multicore CPUs. This allows you to develop your program once, without having to include compile-time conditionals (ifdefs) or special modules for each target with the increased development and maintenance cost.

    Even if you are only interested in GPU-accelerated targets, you can do parallel OpenACC code development and testing on your multicore laptop or workstation without a GPU. This can separate algorithm development from GPU performance tuning. Debugging is often easier on the host than with a heterogeneous binary with both host and device."

------

We will examine this notion about the use of one code file to create different versions of the code, including for the GPU device, in the next example.

The documentation for the pgcc compiler from NVIDIA indicates that the `-ta=multicore` flag is creating OpenMP code to parallelize the loop and that there should be little difference between this and its 'fast' OpenMP version. Run this version and the previous ones to see whether this seems to be the case. (Note that the default of 1 thread is likely best here.)


# 4-openacc-gpu
folder: vector-add/4-openacc-gpu

Examine the Makefile for this version, which is designed to run our vector addition code on the GPU. Note that now we are compiling three versions using the same code file, but with different compiler options. Two versions are for the GPU. Each of these use one of the following pgcc compiler options:

    # initial accelerator options
    AOPTS= -acc -ta=tesla:cc75 -Minfo=accel

    # accelerator options for managed memory
    AOPTSM= -acc -ta=tesla:cc75,managed -Minfo=accel

The -acc means we will generate code for the accelerator GPU card.

The -ta=tesla:cc75 is indicating that the target architecture of our card is the NVIDIA Tesla architecture with compute level 7.5. We've used this because we know that is what our card's architecture is.

Adding the ,managed is saying that the compiler should create CUDA code that uses unified memory (see the CUDA basics activity).

The -Minfo=accel is a useful flag that enables us to see decisions that the compiler is making about how to parallelize the code. This is necessary because we need to be certain that it was able to parallelize the loop to run on many threads on the GPU. 

Note that we also make a third version using pgcc for just the multicore CPU. It uses the compiler options discussed for the previous version.

To see this build the two versions:

    make

When the compiler shows us this:

    17, Loop is parallelizable
         Generating Tesla code
         17, #pragma acc loop gang, vector(128) /* blockIdx.x threadIdx.x */

We can interpret this to mean that blocks of 128 threads will be established as gangs that split the loop on line 17 of the code. The compiler will determine the number of gangs/blocks itself based on the architecture of our particular card. The part in between the /* */ means that it is choosing a 1D grid of 1D blocks, because it is determining a thread's index into the array (the loop index i) based on the blockIdx.x and the threadIdx.x. (Recall the CUDA basics activity where this was done.)

Now let's look at that line 17 of the vectorAdd.c code file and the pragma lines above it. Here's what these pragmas are doing:

The `#pragma acc kernels` directive is indicating that the next block of code should be compiled into a kernel function to be executed on the GPU device. This directive leaves most of the work of creating the GPU device code to the compiler.

The `#pragma acc loop independent` has two important features:

- The `loop` directive is indicating that the compiler should try to parallelize the loop on the GPU inside the kernel function, setting up CUDA blocks of threads and determining ids of threads that match each index into the array.

- The `independent` directive is often necessary for loops like this. The compiler is quite conservative and will decide not to parallelize a loop if it cannot determine whether it is safe from race conditions due to data dependencies. This is a hard task for a compiler, so it often chooses to to parallelize the loop even though you asked for it. So when you add the independent clause, you are signaling to the compiler that you guarantee that the loop has no dependencies that will cause errors. In this case, each thread will be accessing and updating only one data element (one value of array index i for a thread), so we can safely add this clause and let the compiler build the CUDA version.

When the multicore version is compiled, note that the `kernels` pragma directive is ignored and the loop optimization is used because we indicated the -fast flag. In this case, what the pgcc compiler does is use the maximum number of threads available on the CPU that are not hyper-threaded.


# Some Observations and Additional Information

The times you may see will sometimes vary widely. Sometimes the first time is slower than subsequent times, on either the CPU or the GPU. There are any number of reasons for this due to the complexity of our CPUs and operating systems. The GPU takes some time to initialize itself, which adds overhead. Its cores are slower than a typical CPU also. The primary book written for OpenACC so far, called "*Parallel Programming with OpenACC*", edited by Rob Farber, states:

    "The application must perform a million operations per nanosecond (millionths of a second) to match the overhead of starting a parallel section of code on these devices."

Thus, to be effective an OpenACC program on a GPU device must have a fair amount of computation to complete compared to the cost of initializing the device and moving the data to and from the device. This example is not really one of those. Our modern optimizing compilers and operating systems with plenty of cache memory work amazingly well at marching down arrays in memory and performing addition on the CPU, even without using extra threads.

There are several other pragma directives that use close to the same nomenclature as OpenMP. In later examples you will see use of reduction. 

Current versions of the pgcc compiler are remarkably good at generating efficient code for both the multicore CPU and the NVIDIA GPU devices. As an interesting comparison, you could go back to the practice-cuda repo and into the 4-UMVectorAdd-timing folder. Run this example to see how fast it runs using 40 blocks of 128 threads per block:

    ./vectorAdd 128 40

Then compare that to the vector-add/4-openacc-gpu folder's example, where you indicate the same sized vectors using -n:

    ./vectorAdd_acc_managed -n 1048576

