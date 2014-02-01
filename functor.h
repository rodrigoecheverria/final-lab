// MyKernel.h
#ifndef functor_h
#define functor_h
struct plusFunc {
        __host__ __device__ float operator()(float x, float y) const {
        return x + y;
    }
};

#endif
