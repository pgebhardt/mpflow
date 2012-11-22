// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SPARSE_CONJUGATE_HPP
#define FASTEIT_SPARSE_CONJUGATE_HPP

// conjugate class definition
class SparseConjugate {
// constructor and destructor
public:
    SparseConjugate(linalgcuSize_t rows, linalgcuSize_t columns, cudaStream_t stream);
    virtual ~SparseConjugate();

public:
    // solve system
    void solve(linalgcuSparseMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
        linalgcuSize_t iterations, bool dcFree, cudaStream_t stream);

// accessors
public:
    linalgcuSize_t rows() const { return this->mRows; }
    linalgcuSize_t columns() const { return this->mColumns; }

// member
private:
    linalgcuSize_t mRows;
    linalgcuSize_t mColumns;
    linalgcuMatrix_t mResiduum;
    linalgcuMatrix_t mProjection;
    linalgcuMatrix_t mRSOld;
    linalgcuMatrix_t mRSNew;
    linalgcuMatrix_t mTempVector;
    linalgcuMatrix_t mTempNumber;
};

#endif
