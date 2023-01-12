/*
 * @Author: Shengyuan Wang, Kaiyang Yao
 * @Date: 2022-11-17 16:42:44
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:12:53
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/Sequential/floyd_seq.c
 * @Description: Sequential Version Floyd-Warshall
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "MatUtil.h"
#include "../utils/getCommandLine.h"
int main(int argc, char **argv)
{
    int N = 100;
    int numThreads = 1;
    int verbose = 0;
    int experiment = 0;

    //get argument from terminal
    getArguments(argc, argv, &N, &numThreads, &verbose, &experiment);
    if (experiment) {
        verbose = 0;
    }
    if (numThreads != 1){
        numThreads = 1; // not used below
        printf("Warning, this is a sequential version and the number of threads is always 1, even though you used -t\n");
    }
    int *mat = (int*)malloc(sizeof(int)*N*N);
    GenMatrix(mat, N);

    //compute the reference result.
    int *ref = (int*)malloc(sizeof(int)*N*N);
    memcpy(ref, mat, sizeof(int)*N*N);
    double st = omp_get_wtime();
    ST_APSP(ref, N);
    double runtime = omp_get_wtime() - st;

    //print the result
    if (experiment) {
        printf("%f\t", runtime);
    } else {
        printf("Elapsed time (Sequential) = %f s\n", runtime);
    }
    if (verbose) {
        for (int i = 0; i < N*N; i++) {
            printf("%4d", ref[i]);
            if (i % N == N-1) {
                printf("\n");
            }
        }
    }
}
