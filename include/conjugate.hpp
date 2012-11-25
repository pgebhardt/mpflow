// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_CONJUGATE_HPP
#define FASTEIT_CONJUGATE_HPP

// namespace numeric
namespace Numeric {
    // conjugate class definition
    class Conjugate {
    // constructor and destructor
    public:
        Conjugate(fastEIT::dtype::size rows, cublasHandle_t handle, cudaStream_t stream);
        virtual ~Conjugate();

    public:
        // solve system
        void solve(fastEIT::Matrix<fastEIT::dtype::real>* A, fastEIT::Matrix<fastEIT::dtype::real>* x,
            fastEIT::Matrix<fastEIT::dtype::real>* f, fastEIT::dtype::size iterations, cublasHandle_t handle,
            cudaStream_t stream=NULL);

    // helper methods
    public:
        static void addScalar(fastEIT::Matrix<fastEIT::dtype::real>* vector,
            fastEIT::Matrix<fastEIT::dtype::real>* scalar, fastEIT::dtype::size rows,
            fastEIT::dtype::size columns, cudaStream_t stream=NULL);
        static void updateVector(fastEIT::Matrix<fastEIT::dtype::real>* result,
            fastEIT::Matrix<fastEIT::dtype::real>* x1, fastEIT::dtype::real sign,
            fastEIT::Matrix<fastEIT::dtype::real>* x2, fastEIT::Matrix<fastEIT::dtype::real>* r1,
            fastEIT::Matrix<fastEIT::dtype::real>* r2, cudaStream_t stream=NULL);
        static void gemv(fastEIT::Matrix<fastEIT::dtype::real>* result,
            fastEIT::Matrix<fastEIT::dtype::real>* matrix, fastEIT::Matrix<fastEIT::dtype::real>* vector,
            cudaStream_t stream=NULL);

    // accessors
    public:
        fastEIT::dtype::size rows() const { return this->mRows; }

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
        fastEIT::Matrix<fastEIT::dtype::real>* mResiduum;
        fastEIT::Matrix<fastEIT::dtype::real>* mProjection;
        fastEIT::Matrix<fastEIT::dtype::real>* mRSOld;
        fastEIT::Matrix<fastEIT::dtype::real>* mRSNew;
        fastEIT::Matrix<fastEIT::dtype::real>* mTempVector;
        fastEIT::Matrix<fastEIT::dtype::real>* mTempNumber;
    };
}

#endif
