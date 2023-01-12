#!/bin/bash
printf "size  \t50x50 \t71x71 \t100x100 \t141x141 \t200x200 \t282x282 \t400x400 \t564x564 \t800x800 \t1128x1128 \t1600x1600 \t2256x2256 \t3200x3200 \t4512x4512 \t6400x6400\n"

command="cd ./Sequential"
$command
printf "Sequential \t"
for problem_size in 50 71 100 141 200 282 400 564 800 1128 1600 2256 3200 4512 6400
do
    command="./test -n $problem_size -e"
    $command
done
printf "\n"

command="cd ../OpenMP"
$command
printf "OpenMP \t"
for problem_size in 50 71 100 141 200 282 400 564 800 1128 1600 2256 3200 4512 6400
do
    command="./test -n $problem_size -t 8 -e"
    $command
done
printf "\n"

command="cd ../OpenACC"
$command
printf "OpenACC \t"
for problem_size in 50 71 100 141 200 282 400 564 800 1128 1600 2256 3200 4512 6400
do
    command="./test -n $problem_size -e"
    $command
done
printf "\n"

command="cd ../MultiCore"
$command
printf "MultiCore \t"
for problem_size in 50 71 100 141 200 282 400 564 800 1128 1600 2256 3200 4512 6400
do
    command="./test -n $problem_size -t 8 -e"
    $command
done
printf "\n"