#!/bin/bash
###
 # @Author: Shengyuan Wang
 # @Date: 2022-11-21 20:43:27
 # @LastEditors: Shengyuan Wang
 # @LastEditTime: 2022-12-16 12:17:06
 # @FilePath: /445f22/project-team2-shengyuan-kaiyang/OpenMP/run_weak_tests.sh
 # @Description: Weak Scalability Test File
 # 
### 

num_times=$1
num_col=$2
weak_scale_lines=$3

num_row=$num_col
initial_size=$(($num_col*$num_row))
problem_size=$initial_size
double=0
printf "trial  \tproblem size \tprocesses \ttime\n"

for line in $(seq 1 $weak_scale_lines)
do
    echo "line: " $line

    for num_processes in 1 2 4 8 16
    do
        counter=1
        while [ $counter -le $num_times ]
        do
            printf "$counter\t$problem_size\t$num_processes\t"
            command="./floyd_openMP -n $num_col -t $num_processes -e"
            $command
            printf "\n"
            ((counter++))
        done
        
        problem_size=$(($problem_size*2))
        num_col=$(echo $(echo | awk "{print sqrt($problem_size)}") | awk '{print int($0)}')
        num_row=$num_col
    done
    printf "\n"
    double=$((2**$line))
    problem_size=$(($initial_size*$double))
    num_col=$(echo $(echo | awk "{print sqrt($problem_size)}") | awk '{print int($0)}')
    num_row=$num_col
done