#!/bin/bash
num_times=$1
test_obj=$2

printf "trial \t#th \t134217728 \t268435456 \t536870912 \t1073741824 \t2147483616 \n"
# each trial will run num_times using a certain number of threads
for num_threads in 1 2 4 6 8 12 16
do 
    # run the series of problem sizes with the current value of num_threads
    counter=1
    while [ $counter -le $num_times ]
    do
        # $counter is the trial number
        printf "$counter\t$num_threads\t"

        # run each problem size once
        for problem_size in 134217728 268435456 536870912 1073741824 2147483616
        do
            if [ "$num_threads" == "1" ]; then
                command="./countSort_omp -n $problem_size -t 1 -e $test_obj"
            else
                command="./countSort_omp -n $problem_size -t $num_threads -e $test_obj"
            fi
            $command
        done
        printf "\n"
        ((counter++))
    done
    printf "\n"
done
