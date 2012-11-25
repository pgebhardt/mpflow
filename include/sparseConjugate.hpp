// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SPARSE_CONJUGATE_HPP
#define FASTEIT_SPARSE_CONJUGATE_HPP

// namespace numeric
namespace Numeric {
    // conjugate class definition
    class SparseConjugate {
    // constructor and destructor
    public:
        SparseConjugate(fastEIT::dtype::size rows, fastEIT::dtype::size columns,
            cudaStream_t stream=NULL);
        virtual ~SparseConjugate();

    public:
        // solve system
        void solve(fastEIT::SparseMatrix* A, fastEIT::Matrix<fastEIT::dtype::real>* x,
            fastEIT::Matrix<fastEIT::dtype::real>* f, fastEIT::dtype::size iterations, bool dcFree,
            cudaStream_t stream=NULL);

    // accessors
    public:
        fastEIT::dtype::size rows() const { return this->mRows; }
        fastEIT::dtype::size columns() const { return this->mColumns; }

    protected:
        fastEIT::Matrix<fastEIT::dtype::real>* residuum() const { return this->mResiduum; }
        fastEIT::Matrix<fastEIT::dtype::real>* projection() const { return this->mProjection; }
        fastEIT::Matrix<fastEIT::dtype::real>* rsold() const { return this->mRSOld; }
        fastEIT::Matrix<fastEIT::dtype::real>* rsnew() const { return this->mRSNew; }
        fastEIT::Matrix<fastEIT::dtype::real>* tempVector() const { return this->mTempVector; }
        fastEIT::Matrix<fastEIT::dtype::real>* tempNumber() const { return this->mTempNumber; }

    // member
    private:
        fastEIT::dtype::size mRows;
        fastEIT::dtype::size mColumns;
        fastEIT::Matrix<fastEIT::dtype::real>* mResiduum;
        fastEIT::Matrix<fastEIT::dtype::real>* mProjection;
        fastEIT::Matrix<fastEIT::dtype::real>* mRSOld;
        fastEIT::Matrix<fastEIT::dtype::real>* mRSNew;
        fastEIT::Matrix<fastEIT::dtype::real>* mTempVector;
        fastEIT::Matrix<fastEIT::dtype::real>* mTempNumber;
    };
}

#endif
