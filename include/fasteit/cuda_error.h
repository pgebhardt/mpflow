// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// Define this to turn on error checking
// #define CUDA_ERROR_CHECK

#ifndef FASTEIT_INCLUDE_CUDA_ERROR_H
#define FASTEIT_INCLUDE_CUDA_ERROR_H

#define CudaSafeCall(err)   __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()    __cudaCheckError(__FILE__, __LINE__)

#ifndef CUDA_ERROR_CHECK

inline void __cudaSafeCall(cudaError, const char*, const int) { }
inline void __cudaCheckError(const char*, const int) { }

#else

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __cudaCheckError( const char *file, const int line ) {
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    cudaStreamSynchronize(NULL);
    cudaError err = cudaGetLastError();

    if(cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#endif

#endif
