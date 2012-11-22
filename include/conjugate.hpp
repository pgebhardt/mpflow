// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_CONJUGATE_HPP
#define FASTEIT_CONJUGATE_HPP

// conjugate class definition
class Conjugate {
// constructor and destructor
public:
    Conjugate(linalgcuSize_t rows, cublasHandle_t handle, cudaStream_t stream);
    virtual ~Conjugate();

public:
    // solve system
    void solve(linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
        linalgcuSize_t iterations, cublasHandle_t handle, cudaStream_t stream);

// helper methods
public:
    static void add_scalar(linalgcuMatrix_t vector, linalgcuMatrix_t scalar,
        linalgcuSize_t rows, linalgcuSize_t columns, cudaStream_t stream);
    static void update_vector(linalgcuMatrix_t result, linalgcuMatrix_t x1,
        linalgcuMatrixData_t sign, linalgcuMatrix_t x2, linalgcuMatrix_t r1,
        linalgcuMatrix_t r2, cudaStream_t stream);
    static void gemv(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
        linalgcuMatrix_t vector, cudaStream_t);

// accessors
public:
    linalgcuSize_t rows() const { return this->mRows; }

// member
private:
    linalgcuSize_t mRows;
    linalgcuMatrix_t mResiduum;
    linalgcuMatrix_t mProjection;
    linalgcuMatrix_t mRSOld;
    linalgcuMatrix_t mRSNew;
    linalgcuMatrix_t mTempVector;
    linalgcuMatrix_t mTempNumber;
};

#endif
