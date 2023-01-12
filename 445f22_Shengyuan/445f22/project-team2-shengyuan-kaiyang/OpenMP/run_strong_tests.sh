#!/bin/bash
###
 # @Author: Shengyuan Wang
 # @Date: 2022-11-21 18:56:21
 # @LastEditors: Shengyuan Wang
 # @LastEditTime: 2022-12-16 12:16:29
 # @FilePath: /445f22/project-team2-shengyuan-kaiyang/OpenMP/run_strong_tests.sh
 # @Description: Strong Scalability Test File
 # 
### 
num_times=$1

printf "trial \t#th \t1000 \t1400 \t2000 \t2800 \t4000 \n"
# each trial will run num_times using a certain number of threads
for num_processes in 1 2 4 6 8 12 16
do 
    # run the series of problem sizes with the current value of num_threads
    counter=1
    while [ $counter -le $num_times ]
    do
        # $counter is the trial number
        printf "$counter\t$num_processes\t"

        # run each problem size once
        for problem_size in 1000 1400 2000 2800 4000
        do
            if [ "$num_threads" == "1" ]; then
                command="./floyd_openMP -n $problem_size -t 1 -e"
            else
                command="./floyd_openMP -n $problem_size -t $num_processes -e"
            fi
            $command
        done
        printf "\n"
        ((counter++))
    done
    printf "\n"
done