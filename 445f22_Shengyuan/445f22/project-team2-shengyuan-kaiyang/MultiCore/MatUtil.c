/*
 * @Author: Shengyuan Wang
 * @Date: 2022-11-20 12:58:58
 * @LastEditors: Shengyuan Wang
 * @LastEditTime: 2022-12-08 12:58:07
 * @FilePath: /445f22/project-team2-shengyuan-kaiyang/MultiCore/MatUtil.c
 * @Description: Generate Matix and Compare Array
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




