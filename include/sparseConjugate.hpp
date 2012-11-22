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
    SparseConjugate(dtype::size rows, dtype::size columns, cudaStream_t stream);
    virtual ~SparseConjugate();

public:
    // solve system
    void solve(SparseMatrix& A, Matrix<dtype::real>& x, Matrix<dtype::real>& f,
        dtype::size iterations, bool dcFree, cudaStream_t stream);

// accessors
public:
    dtype::size rows() const { return this->mRows; }
    dtype::size columns() const { return this->mColumns; }

// member
private:
    dtype::size mRows;
    dtype::size mColumns;
    Matrix<dtype::real>* mResiduum;
    Matrix<dtype::real>* mProjection;
    Matrix<dtype::real>* mRSOld;
    Matrix<dtype::real>* mRSNew;
    Matrix<dtype::real>* mTempVector;
    Matrix<dtype::real>* mTempNumber;
};

#endif
