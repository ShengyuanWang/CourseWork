<!--
 * @Author: Shengyuan Wang, Kaiyang Yao
 * @Date: 2022-11-08 19:45:27
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 15:22:45
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/README.md
 * @Description: 
 * 
-->

# **Parallel Analysis of Floyd-Warshall algorithm**

## Description
Floyd-Warshall algorithm is a shortest path algorithm named and proposed by a United States famous computer scientist R.W. Floyd in 1962. This paper present the basic concept of this algorithm and its complexity analysis. Then it discusses and compare three different parallel architectures -- OpenMP, OpenACC-GPU, and OpenACC-CPU Multicore -- for decreasing the algorithm complexity.

## Run Versions and Scalability Tests
### Sequential
In `./Sequential`

    make
    ./floyd_seq -n $problem_size -t $num_process -e

### OpenMP
In `./OpenMP`

    make
    ./floyd_openmp -n $problem_size -t $num_processes -e

    bash run_strong_tests.sh $repeat_time
    bash run_weak_tests.sh $repeat_time $start_size $num_line


### OpenACC-Multicore
In `./MultiCore`

    make
    ./floyd_mc -n $problem_size -t $num_processes -e
    
    bash run_strong_tests.sh $repeat_time
    bash run_weak_tests.sh $repeat_time $start_size $num_line

### OpenACC-GPU
In `./OpenACC`

    make
    ./floyd_acc -n $problem_size -t $num_processes -e


## SpeedUp Tests

  MultiCore

    ./test -n 2000 -t 8 -e
    ./test -n 2828 -t 8 -e
    ./test -n 4000 -t 8 -e

  OpenACC

    ./test -n 2000 -e
    ./test -n 2828 -e
    ./test -n 4000 -e

  OpenMP

    ./test -n 2000 -e
    ./test -n 2828 -e
    ./test -n 4000 -e

## Authors

  - **Shengyuan Wang** -
    [swang3@macalester.edu](swang3@macalester.edu)
  - **Kaiyang Yao** -
    [kyao@macalester.edu](kyao@macalester.edu)

