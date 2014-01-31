#include <iostream>
#include <fstream>
#include <cuda.h>
#include <math.h>

#define TILE_DIM 4

struct sigmoidFunc {
        __host__ __device__ float operator()(float z) const {
        return 1.0/(1.0 + exp(-(z/1000)));
    }
};  

//TRY TO PUT A SIGMOID FUNCTOR HERE !!!!
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
    int i, N =6, M = 3;
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
