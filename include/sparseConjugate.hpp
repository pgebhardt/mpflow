// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SPARSE_CONJUGATE_HPP
#define FASTEIT_SPARSE_CONJUGATE_HPP

// namespace numeric
namespace numeric {
    // conjugate class definition
    class SparseConjugate {
    // constructor and destructor
    public:
        SparseConjugate(dtype::size rows, dtype::size columns,
            cudaStream_t stream=NULL);
        virtual ~SparseConjugate();

    public:
        // solve system
        void solve(SparseMatrix* A, Matrix<dtype::real>* x,
            Matrix<dtype::real>* f, dtype::size iterations, bool dcFree,
            cudaStream_t stream=NULL);

    // accessors
    public:
        dtype::size rows() const { return this->mRows; }
        dtype::size columns() const { return this->mColumns; }

    protected:
        Matrix<dtype::real>* residuum() const { return this->mResiduum; }
        Matrix<dtype::real>* projection() const { return this->mProjection; }
        Matrix<dtype::real>* rsold() const { return this->mRSOld; }
        Matrix<dtype::real>* rsnew() const { return this->mRSNew; }
        Matrix<dtype::real>* tempVector() const { return this->mTempVector; }
        Matrix<dtype::real>* tempNumber() const { return this->mTempNumber; }

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
}

#endif
