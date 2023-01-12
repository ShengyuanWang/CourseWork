/*
 * @Author: Shengyuan Wang, Kaiyang Yao
 * @Date: 2022-11-19 18:56:40
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:43:06
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/OpenMP/MatUtil.c
 * @Description: Generate Matrix and Compare Array
 * 
 */
#include "MatUtil.h"

void GenMatrix(int *mat, const int N)
{
	for(int i = 0; i < N*N; i ++)
		mat[i] = rand()%32 - 1;
	for(int i = 0; i < N; i++)
		mat[i*N + i] = 0;

}

bool CmpArray(const int *l, const int *r, const int eleNum)
{
	for(int i = 0; i < eleNum; i ++)
		if(l[i] != r[i])
		{
			printf("ERROR: l[%d] = %d, r[%d] = %d\n", i, l[i], i, r[i]);
			return false;
		}
	return true;
}


