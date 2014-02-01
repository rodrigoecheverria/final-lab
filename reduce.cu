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
__global__ void ZipMapReduceKernel(float* X, float* Y, float* R, int size, 
                                    MapFunction mapFunction, float neutralElement, 
                                    ReduceFunction reduceFunction)
{
    __shared__ float sX[TILE_DIM];
    __shared__ float sY[TILE_DIM];
    __shared__ float sR[TILE_DIM];
  
    unsigned int i = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
        
    //Load data from memory to shared memory collectively
    sX[tid] = X[i];
    sY[tid] = Y[i];
    sR[tid] = neutralElement;
    __syncthreads();
    
    //Zip and Map: sR <- Map(Zip(sX,sY))
    sR[tid] = mapFunction(sX[tid],sY[tid]);
    __syncthreads();
    
    //Reduce
    for(unsigned int s = TILE_DIM / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sR[tid] = reduceFunction(sR[tid], sR[tid + s]);
        __syncthreads();
    }
    
    //Delegate (thread 0) writes to memory
    if (tid == 0)
        R[bid] = sR[0];
}

template<typename MapFunction,
         typename ReduceFunction>
float ZipMapReduce(float* d_X, float* d_Y, int size, MapFunction mapFunction, 
                   float neutralElement, ReduceFunction reduceFunction)
{
    float *R, *d_R, r = neutralElement;
    dim3 dimBlock(TILE_DIM);
    dim3 dimGrid((size + dimBlock.x - 1) / dimBlock.x);
    
    //Create auxiliary vector
    R = (float *) malloc (sizeof(float) * dimGrid.x);
    cudaMalloc((void **)&d_R, sizeof(float) * dimGrid.x);
    
    //Reduce to vector R
	ZipMapReduceKernel<<<dimGrid, dimBlock>>>(d_X, d_Y, d_R, size, mapFunction, neutralElement, reduceFunction);
    cudaThreadSynchronize();
    cudaMemcpy(R, d_R, dimGrid.x * sizeof(float), cudaMemcpyDeviceToHost);

    //Reduce remaining values in host
    for (int i = 0; i < dimGrid.x; i++)
        r = reduceFunction(r, R[i]);
    cudaFree(d_R);
    free(R);
    return r;
}
// Invoke kernel
int main(int argc, char *argv[])
{
    float *A, *B, *d_A, *d_B, C;
    int i, N =6, M = 6;
    
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
    
    C = ZipMapReduce(d_A, d_B, M*N, mapFunction, 0.0, reduceFunction);
    printf("Result: %f\n",C);
 
    cudaFree(d_A); cudaFree(d_B);
    free(A); free(B);
    
   
    return 0;
    
}
