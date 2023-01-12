#!/bin/bash
num_times=$1
initial_size=$2
weak_scale_lines=$3
test_obj=$4
printf "trial \tproblem size \tthreads \ttime\n"
problem_size=$initial_size
double=0

for line in $(seq 1 $weak_scale_lines)
do
    echo "line: " $line

    for num_threads in 1 2 4 8 16
    do
        counter=1
        while [ $counter -le $num_times ]
        do
            printf "$counter\t$problem_size\t$num_threads\t"
            command="./countSort_omp -n $problem_size -t $num_threads -e $test_obj"
            $command
            printf "\n"
            ((counter++))
        done
        problem_size=$(($problem_size*2))
    done
    printf "\n"
    double=$((2**$line))
    problem_size=$(($initial_size*$double))
done