#include <iostream>
#include <fstream>
#include <cuda.h>
#include <math.h>

#define TILE_DIM 4

struct plusFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x + y;
    }
};
struct prodFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x * y;
    }
};

template<typename MapFunction,
         typename ReduceFunction>
__global__ void ZipMapKernel(float* X, float* Y, float* R, int size, 
                                    MapFunction mapFunction)
{
    __shared__ float sX[TILE_DIM];
    __shared__ float sY[TILE_DIM];
    //__shared__ float sR[TILE_DIM];
  
    unsigned int i = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int tid = threadIdx.x;
        
    //Load data from memory to shared memory collectively
    sX[tid] = X[i];
    sY[tid] = Y[i];
    __syncthreads();
    
    //Zip and Map: sR <- Map(Zip(sX,sY))
    if (i < size) 
        R[tid] = mapFunction(sX[tid],sY[tid]);
    __syncthreads();
}

template<typename MapFunction,
         typename ReduceFunction>
float ZipMap(float* d_X, float* d_Y, int size, MapFunction mapFunction)
{
    float *R, *d_R;
    dim3 dimBlock(TILE_DIM);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    
    //Create auxiliary vector
    R = (float *) malloc (sizeof(float) * size);
    cudaMalloc((void **)&d_R, sizeof(float) * size);
    
    //Reduce to vector R
	ZipMapReduceKernel<<<dimGrid, dimBlock>>>(d_X, d_Y, d_R, size, mapFunction);
    cudaThreadSynchronize();
    cudaMemcpy(R, d_R, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);
    
    return r;
}
// Invoke kernel
int main(int argc, char *argv[])
{
    float *A, *B, *d_A, *d_B, C;
    int i, N =2, M = 2;
    
    A = (float *) malloc (sizeof(float) * M * N);
    B = (float *) malloc (sizeof(float) * M * N);
    cudaMalloc((void **)&d_A, M * N * sizeof(float));
    cudaMalloc((void **)&d_B, M * N * sizeof(float));
    for (i = 0; i < N * M; i++) A[i] = i;
    for (i = 0; i < M * N; i++) B[i] = i;
    cudaMemcpy(d_A,	A, N * M * sizeof(float), cudaMemcpyHostToDevice);	
    cudaMemcpy(d_B,	B, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    prodFunc mapFunction;
    plusFunc reduceFunction;
    
    C = ZipMapReduce(d_A + N, d_A + N, (M-1)*N, mapFunction, 0.0, reduceFunction);
    printf("Result: %f\n",C);
 
    cudaFree(d_A); cudaFree(d_B);
    free(A); free(B);
    
   
    return 0;
    
}
