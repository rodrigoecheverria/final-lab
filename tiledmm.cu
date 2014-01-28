#include <cuda.h>
#include <stdio.h>
#define TILE_DIM 16

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, 
    int BRows, int BCols, int CRows, int CCols) 
{
    float CValue = 0;
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++)
    {
         if (k * TILE_DIM + threadIdx.x < ACols && Row < ARows)   
            As[threadIdx.y][threadIdx.x] = 
                A[Row * ACols + k * TILE_DIM + threadIdx.x];
         else                                                 
            As[threadIdx.y][threadIdx.x] = 0.0;

         if (k * TILE_DIM + threadIdx.y < BRows && Col < BCols)   
            Bs[threadIdx.y][threadIdx.x] = 
                B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
         else      
            Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n) 
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols) 
        C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + 
            (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}


// Invoke kernel
int main(int argc, char *argv[])
{
    float *d_A, *d_B, *d_C, *A, *B, *C;
    int i, N = 32;
    A = (float *) malloc (sizeof(float) * N * N);
    B = (float *) malloc (sizeof(float) * N * N);
    C = (float *) malloc (sizeof(float) * N * N);
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));
    
    for (i = 0; i < N * N; i++)
    {
        A[i] = 1.0; B[i] = 1.0; C[i] = 0.0;
    }
    
    cudaMemcpy(d_A,	A, N * N * sizeof(float), cudaMemcpyHostToDevice);	
    cudaMemcpy(d_B,	B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    MatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C,N,N,N,N,N,N);
    cudaThreadSynchronize();
    
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    for (i = 0; i < N * N; i++)
    {
        printf("%f, ",C[i]);
    }
    return 0;
    
}
