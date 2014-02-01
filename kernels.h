// kernels.h
#ifndef kernels_h
#define kernels_h
template<typename UnaryFunction>
__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, 
    int BRows, int BCols, int CRows, int CCols, bool addBias, UnaryFunction activationFunction );
#endif
