/*
 * @Author: Shengyuan Wang, Kaiyang Yao
 * @Date: 2022-11-20 12:58:58
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:56:37
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/MultiCore/Floyd_mc.c
 * @Description: Floyd Marshall Algorithm OpenACC-Multicore Version
 * 
 */
#include <string.h>
#include <stdlib.h>

#include "Floyd_mc.h"

void Floyd_Warshall(int* matrix, int size) {
    int *row_k = (int*)malloc(sizeof(int)*size);

    for (int k = 0; k < size; k++) {
        memcpy(row_k, matrix + (k * size), sizeof(int)*size);
        #pragma acc parallel loop
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
