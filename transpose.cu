#include <iostream>
#include <fstream>
#include <cuda.h>

#define TILE_DIM 64
#define BLOCK_ROWS 8

__global__ void TransposeKernel(float *d_A, float *d_At, int cols, int rows)
{
    __shared__ float block[TILE_DIM][TILE_DIM+1];

    // read the matrix tile into shared memory
    unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    if((xIndex < cols) && (yIndex < rows))
    {
            unsigned int index_in = yIndex * cols + xIndex;
            block[threadIdx.y][threadIdx.x] = d_A[index_in];
    }

    __syncthreads();

    // write the transposed matrix tile to global memory
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    if((xIndex < rows) && (yIndex < cols))
    {
            unsigned int index_out = yIndex * rows + xIndex;
            d_At[index_out] = block[threadIdx.x][threadIdx.y];
    }  
}

void Transpose(float* d_A, float* d_B, int cols, int rows)
{
    dim3 dimBlock(TILE_DIM, TILE_DIM,1);
    dim3 dimGrid(( cols + TILE_DIM -1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM,1);
	TransposeKernel<<<dimGrid, dimBlock>>>(d_A, d_B, cols, rows);
    cudaThreadSynchronize();
}

// Invoke kernel
int main(int argc, char *argv[])
{
    float *d_A, *d_B, *A, *B;
    int i, N = 3, M = 4;
    A = (float *) malloc (sizeof(float) * N * M);
    B = (float *) malloc (sizeof(float) * N * M);
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, M * N * sizeof(float));
    
    for (i = 0; i < N * M; i++) A[i] =i;
	
    cudaMemcpy(d_A,	A, N * M * sizeof(float), cudaMemcpyHostToDevice);
   
	Transpose(d_A, d_B, M, N);
    
    cudaMemcpy(A, d_A, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A); cudaFree(d_B);
    
    printf("A:\n");
    for (i = 0; i < N * M; i++)
    {
			if ((i % M) == 0) printf("\n");
			printf("%f, ",A[i]);
    }
    
    printf("B:\n");
    for (i = 0; i < N * M; i++)
    {
			if ((i % N) == 0) printf("\n");
			printf("%f, ",B[i]);
    }
    return 0;
}
