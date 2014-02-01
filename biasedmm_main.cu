#include <iostream>
#include <fstream>
#include "kernels.h"
#include "constants.h"

struct sigmoidFunc {
        __host__ __device__ float operator()(float z) const {
        return 1.0/(1.0 + exp(-(z)));
    }
}; 

// Invoke kernel
int main(int argc, char *argv[])
{
    float *d_A, *d_B, *d_C, *A, *B, *C;
    int i, N =600, M = 300;
    A = (float *) malloc (sizeof(float) * M * N);
    B = (float *) malloc (sizeof(float) * M * (N+1));
    C = (float *) malloc (sizeof(float) * M * M);
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, M * (N+1) * sizeof(float));
    cudaMalloc((void **)&d_C, M * M * sizeof(float));
    
    for (i = 0; i < N * M; i++) A[i] =i;
    for (i = 0; i < M * (N+1); i++) B[i] = i;
    for (i = 0; i < M * M; i++) C[i] = 0.0;
	
    cudaMemcpy(d_A,	A, N * M * sizeof(float), cudaMemcpyHostToDevice);	
    cudaMemcpy(d_B,	B, M * (N + 1) * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((M + dimBlock.x -1) / dimBlock.x, (M  + dimBlock.y -1) / dimBlock.y);
    sigmoidFunc sigmoidf;
		MatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,M,N,N,M,M,M,true,sigmoidf);
    cudaThreadSynchronize();
    
    cudaMemcpy(C, d_C, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    for (i = 0; i < M * M; i++)
    {
			if ((i % N) == 0) printf("\n");
			printf("%f, ",C[i]);
    }
    return 0;
}
