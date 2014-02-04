#include <iostream>
#include <fstream>
#include <cuda.h>
#include <math.h>

#define TILE_DIM 4

struct plusOneFunc {
        __host__ __device__ float operator()(float z) const {
        return z+1;
    }
};
template<typename UnaryFunction>
__global__ void Tranform(float* Y, int Rows, int Cols, UnaryFunction activationFunction)
{
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    if (Row < Rows && Col < Cols) 
        Y[((blockIdx.y * blockDim.y + threadIdx.y) * Cols) + 
            (blockIdx.x * blockDim.x) + threadIdx.x] = 
            activationFunction(Y[((blockIdx.y * blockDim.y + threadIdx.y) * Cols) + 
            (blockIdx.x * blockDim.x) + threadIdx.x]);
}

// Invoke kernel
int main(int argc, char *argv[])
{
    float *d_A, *d_B, *d_C, *A, *B, *C;
    int i, N =6, M = 6;
    A = (float *) malloc (sizeof(float) * M * N);
    B = (float *) malloc (sizeof(float) * M * N);
    C = (float *) malloc (sizeof(float) * M * M);
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, M * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * M * sizeof(float));
    
    for (i = 0; i < N * M; i++) A[i] =i;
    for (i = 0; i < M * N; i++) B[i] = i;
    for (i = 0; i < M * M; i++) C[i] = 0.0;
	
    cudaMemcpy(d_A,	A, N * M * sizeof(float), cudaMemcpyHostToDevice);	
    cudaMemcpy(d_B,	B, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((N + dimBlock.x -1) / dimBlock.x, (M  + dimBlock.y -1) / dimBlock.y);
    plusOneFunc sigmoidf;
		Tranform<<<dimGrid, dimBlock>>>(d_A,N,M,sigmoidf);
    cudaThreadSynchronize();
    
    cudaMemcpy(A, d_A, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    for (i = 0; i < N * M; i++)
    {
			printf("%f, ",A[i]);
			printf("\n");
    }
    return 0;
    
}
