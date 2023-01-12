/*
 * @Author: Shengyuan Wang
 * @Date: 2022-11-19 19:57:28
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:52:25
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/OpenACC/MatUtil.c
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


