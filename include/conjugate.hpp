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
    Conjugate(dtype::size rows, cublasHandle_t handle, cudaStream_t stream);
    virtual ~Conjugate();

public:
    // solve system
    void solve(linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
        dtype::size iterations, cublasHandle_t handle, cudaStream_t stream);

// helper methods
public:
    static void add_scalar(linalgcuMatrix_t vector, linalgcuMatrix_t scalar,
        dtype::size rows, dtype::size columns, cudaStream_t stream);
    static void update_vector(linalgcuMatrix_t result, linalgcuMatrix_t x1,
        dtype::real sign, linalgcuMatrix_t x2, linalgcuMatrix_t r1,
        linalgcuMatrix_t r2, cudaStream_t stream);
    static void gemv(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
        linalgcuMatrix_t vector, cudaStream_t);

// accessors
public:
    dtype::size rows() const { return this->mRows; }

// member
private:
    dtype::size mRows;
    linalgcuMatrix_t mResiduum;
    linalgcuMatrix_t mProjection;
    linalgcuMatrix_t mRSOld;
    linalgcuMatrix_t mRSNew;
    linalgcuMatrix_t mTempVector;
    linalgcuMatrix_t mTempNumber;
};

#endif
