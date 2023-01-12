/*
 * @Author: Shengyuan Wang
 * @Date: 2022-11-19 19:57:28
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 15:47:15
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/OpenACC/Floyd_acc.c
 * @Description: Floyd Warshall Algorithm OpenACC-GPU Version
 * 
 */
#include <omp.h>
#include <string.h>
#include <stdlib.h>

#include "Floyd_acc.h"

void Floyd_Warshall(int* matrix, int size) {
    int *row_k = (int*)malloc(sizeof(int)*size);

    // parallel here
    #pragma acc kernels
    #pragma acc loop independent
    for (int k = 0; k < size; k++) {
        memcpy(row_k, matrix + (k * size), sizeof(int)*size);
        #pragma acc loop independent
        for (int i = 0; i < size; ++i) {
            #pragma acc loop independent
            for (int j = 0; j < size; ++j) {
                if (matrix[i * size + k] != -1 && row_k[j] != -1) {
                    int new_path = matrix[i * size + k] + row_k[j];
                    if (new_path < matrix[i * size + j] || matrix[i * size + j] == -1)
                        matrix[i * size + j] = new_path;
                }
            }
        }
    }
}
