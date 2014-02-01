#include <iostream>
#include <fstream>
#include <cuda.h>
#include <math.h>
#define TILE_DIM 8

struct sigmoidFunc {
        __host__ __device__ float operator()(float z) const {
        return (1.0 + z);
    }
}; 

template<typename UnaryFunction>
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, 
    int BRows, int BCols, int CRows, int CCols, bool addBias, UnaryFunction activationFunction ) 
{
    float CValue = 0;
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;
    int biasOffset = addBias ? 1 : 0; 
	
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++)           //floor(ACols/TILE_DIM)
    {
         if (k * TILE_DIM + threadIdx.x < ACols && Row < ARows)   
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * TILE_DIM + threadIdx.x];
         else                                                 
            As[threadIdx.y][threadIdx.x] = 0.0;

         if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols)   
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y + biasOffset) * BCols + Col]; //+1 one row if bias
         else      
            Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) 
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }
    if (addBias)
	{
		__shared__ float BiasRow[TILE_DIM];
	    
		if (threadIdx.y == 0){
		  if (Col < BCols){
			BiasRow[threadIdx.x] = B[Col];
		  }else{
		  	BiasRow[threadIdx.x] = 0.0;
			}
			}
	    __syncthreads();
		
		CValue += BiasRow[threadIdx.x];
		
		__syncthreads();
		
	}
	
    if (Row < CRows && Col < CCols) 
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + 
            (blockIdx.x * blockDim.x) + threadIdx.x] = 
            activationFunction(CValue);
}

// Invoke kernel
int main(int argc, char *argv[])
{
    float *d_A, *d_B, *d_C, *A, *B, *C;
    int i, N = 3, M = 4, O = 25;
    A = (float *) malloc (sizeof(float) * N * M);
    B = (float *) malloc (sizeof(float) * (M+1) * O);
    C = (float *) malloc (sizeof(float) * N * O);
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, (M+1) * O * sizeof(float));
    cudaMalloc((void **)&d_C, N * O * sizeof(float));
    
    for (i = 0; i < N * M; i++) A[i] =i;
    for (i = 0; i < (M+1) * O; i++) B[i] = i;
    for (i = 0; i < M * O; i++) C[i] = 0.0;
	
    cudaMemcpy(d_A,	A, N * M * sizeof(float), cudaMemcpyHostToDevice);	
    cudaMemcpy(d_B,	B, (M+1) * O * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((O + dimBlock.x -1) / dimBlock.x, (N  + dimBlock.y -1) / dimBlock.y);
    sigmoidFunc sigmoidf;
		MatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,N,M,M,O,N,O,true,sigmoidf);
    cudaThreadSynchronize();
    
    cudaMemcpy(C, d_C, N * O * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    for (i = 0; i < N * O; i++)
    {
			if ((i % O) == 0) printf("\n");
			printf("%f, ",C[i]);
    }
    return 0;
}
