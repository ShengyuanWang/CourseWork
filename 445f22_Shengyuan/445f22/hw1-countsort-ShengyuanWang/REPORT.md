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
# HW1 Report
### `Shengyuan Wang`

## Exploration of Sequential version
When I test it with a small string, like
```
make countSort_seq
./countSort_seq -n 20 -v
```
The result turn out to be
```
number of chars in input: 20
generate input: 0.000008 seconds
input: n+Rdp:N((gtjd,t&@>"\
generate counts: 0.000001 seconds
generate output: 0.000003 seconds
output: "&((+,:>@NR\ddgjnptt
```
It appears that the string is being sorted according to the ascii character values.
```
./countSort_seq -n 40 -v
```
```
number of chars in input: 40
generate input: 0.000008 seconds
input: n+Rdp:N((gtjd,t&@>"\ 8h%AM{d3j6H'_iioG6,
generate counts: 0.000001 seconds
generate output: 0.000004 seconds
output:  "%&'((+,,3668:>@AGHMNR\_dddghiijjnoptt{
```


Then I try the sequential version with a larger problem size using my shell scripts

```
#!/bin/bash
START=20
START_NUM=1048576
echo "===== Testing Profiling code for Sequential Version ===="
while(( ${START}<=30 ))
do
    echo "Using power of ${START}"
    ./countSort_seq -n ${START_NUM}
    echo "------------------------"
    let "START++"
    START_NUM=`expr ${START_NUM} \* 2`
done
```
And result turn out to be
```
===== Testing Profiling code for Sequential Version ====
Using power of 20
generate input: 0.021424 seconds
generate counts: 0.008087 seconds
generate output: 0.007577 seconds
------------------------
Using power of 21
generate input: 0.029040 seconds
generate counts: 0.011078 seconds
generate output: 0.010336 seconds
------------------------
Using power of 22
generate input: 0.041916 seconds
generate counts: 0.016407 seconds
generate output: 0.015034 seconds
------------------------
Using power of 23
generate input: 0.058499 seconds
generate counts: 0.022894 seconds
generate output: 0.021367 seconds
------------------------
Using power of 24
generate input: 0.095663 seconds
generate counts: 0.043075 seconds
generate output: 0.042095 seconds
------------------------
Using power of 25
generate input: 0.192868 seconds
generate counts: 0.085062 seconds
generate output: 0.084759 seconds
------------------------
Using power of 26
generate input: 0.381160 seconds
generate counts: 0.170888 seconds
generate output: 0.168443 seconds
------------------------
Using power of 27
generate input: 0.764506 seconds
generate counts: 0.341144 seconds
generate output: 0.336335 seconds
------------------------
Using power of 28
generate input: 1.638598 seconds
generate counts: 0.685692 seconds
generate output: 0.676912 seconds
------------------------
Using power of 29
generate input: 3.175629 seconds
generate counts: 1.422833 seconds
generate output: 1.332597 seconds
------------------------
Using power of 30
generate input: 6.201753 seconds
generate counts: 2.728440 seconds
generate output: 2.691523 seconds
------------------------
```

## Making my own OpenMP version
First, I make a copy of Sequential version and change the getArgument function and timing functions.
