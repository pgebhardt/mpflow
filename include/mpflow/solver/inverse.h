// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLDUE_SOLVER_INVERSE_H
#define MPFLOW_INCLDUE_SOLVER_INVERSE_H

namespace mpFlow {
namespace solver {
    // inverse solver class definition
    template <
        class dataType,
        template <class> class numericalSolverType
    >
    class Inverse {
    public:
        enum RegularizationType {
            identity,
            diagonal,
            totalVariational
        };

        // constructor
        Inverse(std::shared_ptr<numeric::IrregularMesh const> const mesh,
            std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
            unsigned const parallelImages, cublasHandle_t const handle,
            cudaStream_t const stream);

    public:
        // update jacobian matrix and recalculated all intermediate matrices
        void updateJacobian(std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
            cublasHandle_t const handle, cudaStream_t const stream);
            
        // inverse solving
        unsigned solve(std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
            std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
            unsigned const steps, cublasHandle_t const handle, cudaStream_t const stream,
            std::shared_ptr<numeric::Matrix<dataType>> result);
            
    private:
        // update intermediate matrices
        void calcRegularizationMatrix(cublasHandle_t const handle, cudaStream_t const stream);
        void calcJacobianSquare(cublasHandle_t const handle, cudaStream_t const stream);
        void calcExcitation(std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
            std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
            cublasHandle_t const handle, cudaStream_t const stream);

    public:
        // accessors for regularization parameter
        void setRegularizationFactor(dataType const factor, cublasHandle_t const handle,
            cudaStream_t const stream) {
            this->regularizationFactor_ = factor;
            this->calcRegularizationMatrix(handle, stream);
        }
        void setRegularizationType(RegularizationType const type, cublasHandle_t const handle,
            cudaStream_t const stream) {
            this->regularizationType_ = type;
            this->calcRegularizationMatrix(handle, stream);
        }
        void setRegularizationParameter(dataType const factor, RegularizationType const type,
            cublasHandle_t const handle, cudaStream_t const stream) {
            this->regularizationFactor_ = factor;
            this->regularizationType_ = type;
            this->calcRegularizationMatrix(handle, stream);
        }
        
        dataType regularizationFactor() const { return this->regularizationFactor_; }
        RegularizationType regularizationType() const { return this->regularizationType_; }
        
    private:
        // member
        dataType regularizationFactor_;
        RegularizationType regularizationType_;
        std::shared_ptr<numericalSolverType<dataType>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dataType>> difference;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType> const> jacobian;
        std::shared_ptr<numeric::Matrix<dataType>> jacobianSquare;
        std::shared_ptr<numeric::Matrix<dataType>> regularizationMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
    };
}
}

#endif
