/*
 * @Author: Shengyuan Wang, Kaiyang Yao
 * @Date: 2022-11-19 18:56:40
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:42:56
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/OpenMP/Floyd_openMP.c
 * @Description: Floyd Warshall algorithm OpenMP Version
 * 
 */
#include <omp.h>
#include <string.h>
#include <stdlib.h>

#include "Floyd_openMP.h"

void Floyd_Warshall(int* matrix, int size) {
    int *row_k = (int*)malloc(sizeof(int)*size);
    //parallel here
    #pragma omp parallel \
    shared(row_k)
    for (int k = 0; k < size; k++) {
        #pragma omp master
        memcpy(row_k, matrix + (k * size), sizeof(int)*size);
        #pragma omp for schedule(static)
        for (int i = 0; i < size; ++i) {
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
