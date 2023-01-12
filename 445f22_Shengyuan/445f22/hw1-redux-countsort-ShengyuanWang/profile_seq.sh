#!/bin/bash
printf "problem_size \t input \t counts \t output \n"
for problem_size in 1048576 2097152 4194304 8388608 16777216 33664432 67108864 134217728 268435456 536870912 1073741824
do
    printf "$problem_size \t"
    command="./countSort_seq -n $problem_size -e"
    $command
done
printf "\n"