/*
 * @Author: Shengyuan Wang
 * @Date: 2022-11-20 12:58:58
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:58:12
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/MultiCore/MatUtil.h
 * @Description: Generate Matrix and Compare Array
 * 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

/// Generate a random matrix.
//
// Parameters:
// int *mat - pointer to the generated matrix. mat should have been 
//            allocated before callling this function.
// const int N - number of vertices.
void GenMatrix(int *mat, const int N);

/// Compare the content of two integer arrays. Return true if they are
// exactly the same; otherwise return false.
//
// Parameters:
// const int *l, const int *r - the two integer arrays to be compared.
// const int eleNum - the length of the two matrices.
bool CmpArray(const int *l, const int *r, const int eleNum);



