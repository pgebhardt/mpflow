// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <vector>
#include <array>
#include <tuple>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/sparse_matrix.h"
#include "../include/mesh.h"
#include "../include/electrodes.h"
#include "../include/basis.h"
#include "../include/model.h"
#include "../include/conjugate.h"
#include "../include/sparse_conjugate.h"
#include "../include/forward.h"
#include "../include/forward_cuda.h"

// create forward_solver
template <
    class BasisFunction,
    class NumericSolver
>
fastEIT::ForwardSolver<BasisFunction, NumericSolver>::ForwardSolver(
    Mesh<BasisFunction>* mesh, Electrodes* electrodes,
    const Matrix<dtype::real>& measurment_pattern, const Matrix<dtype::real>& drive_pattern,
    dtype::real sigma_ref, dtype::size num_harmonics, cublasHandle_t handle,
    cudaStream_t stream)
    : model_(NULL), numeric_solver_(NULL), drive_count_(drive_pattern.columns()),
        measurment_count_(measurment_pattern.columns()), jacobian_(NULL), voltage_(NULL),
        voltage_calculation_(NULL), elemental_jacobian_matrix_(NULL) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("ForwardSolver::ForwardSolver: handle == NULL");
    }

    // create model
    this->model_ = new Model<BasisFunction>(mesh, electrodes, sigma_ref, num_harmonics, handle,
        stream);

    // create NumericSolver solver
    this->numeric_solver_ = new NumericSolver(mesh->nodes().rows(),
        this->drive_count() + this->measurment_count(), stream);

    // create matrices
    this->jacobian_ = new Matrix<dtype::real>(measurment_pattern.data_columns() *
        drive_pattern.data_columns(), mesh->elements().rows(), stream);
    this->voltage_  = new Matrix<dtype::real>(this->measurment_count(), this->drive_count(), stream);
    this->voltage_calculation_  = new Matrix<dtype::real>(this->measurment_count(),
        mesh->nodes().rows(), stream);
    this->elemental_jacobian_matrix_  = new Matrix<dtype::real>(mesh->elements().rows(),
        Matrix<dtype::real>::block_size, stream);

    // create matrices
    for (dtype::index harmonic = 0; harmonic < num_harmonics + 1; ++harmonic) {
        this->phi().push_back(new Matrix<dtype::real>(mesh->nodes().rows(),
            this->drive_count() + this->measurment_count(), stream));
        this->excitation().push_back(new Matrix<dtype::real>(mesh->nodes().rows(),
            this->drive_count() + this->measurment_count(), stream));
    }

    // create pattern matrix
    Matrix<dtype::real> pattern(drive_pattern.rows(),
        this->drive_count() + this->measurment_count(), stream);

    // fill pattern matrix with drive pattern
    for (dtype::index row = 0; row < pattern.rows(); ++row) {
        for (dtype::index column = 0; column < this->drive_count(); ++column) {
            pattern(row, column) = drive_pattern(row, column);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::index row = 0; row < pattern.rows(); ++row) {
        for (dtype::index column = 0; column < this->measurment_count(); ++column) {
            pattern(row, column + this->drive_count()) = measurment_pattern(row, column);
        }
    }
    pattern.copyToDevice(stream);

    // calc excitation components
    this->model().calcExcitationComponents(pattern, handle, stream, &this->excitation());

    // calc voltage calculation matrix
    dtype::real alpha = -1.0f, beta = 0.0f;

    // one prerun for cublas
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurment_pattern.data_columns(),
        this->model().excitation_matrix().data_rows(), measurment_pattern.data_rows(), &alpha,
        measurment_pattern.device_data(), measurment_pattern.data_rows(),
        this->model().excitation_matrix().device_data(), this->model().excitation_matrix().data_rows(),
        &beta, this->voltage_calculation().device_data(), this->voltage_calculation().data_rows());

    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurment_pattern.data_columns(),
        this->model().excitation_matrix().data_rows(), measurment_pattern.data_rows(), &alpha,
        measurment_pattern.device_data(), measurment_pattern.data_rows(),
        this->model().excitation_matrix().device_data(), this->model().excitation_matrix().data_rows(),
        &beta, this->voltage_calculation().device_data(), this->voltage_calculation().data_rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("ForwardSolver::ForwardSolver: calc voltage calculation");
    }

    // init jacobian calculation matrix
    this->initJacobianCalculationMatrix(handle, stream);
}

// release solver
template <
    class BasisFunction,
    class NumericSolver
>
fastEIT::ForwardSolver<BasisFunction, NumericSolver>::~ForwardSolver() {
    // cleanup
    delete this->jacobian_;
    delete this->voltage_;
    delete this->voltage_calculation_;
    delete this->elemental_jacobian_matrix_;

    for (auto phi : this->phi()) {
        delete phi;
    }
    for (auto excitation : this->excitation()) {
        delete excitation;
    }
    delete this->model_;
    delete this->numeric_solver_;
}

// init jacobian calculation matrix
template <
    class BasisFunction,
    class NumericSolver
>
void fastEIT::ForwardSolver<BasisFunction, NumericSolver>::initJacobianCalculationMatrix(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("ForwardSolver::initJacobianCalculationMatrix: handle == NULL");
    }

    // variables
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;
    std::array<BasisFunction*, BasisFunction::nodes_per_element> basis_functions;

    // fill connectivity and elementalJacobianMatrix
    for (dtype::index element = 0; element < this->model().mesh().elements().rows(); ++element) {
        // get element indices
        indices = this->model().mesh().elementIndices(element);

        // calc corresponding basis functions
        for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
            basis_functions[node] = new BasisFunction(
                this->model().mesh().elementNodes(element), node);
        }

        // fill matrix
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; ++i) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; ++j) {
                // set elementalJacobianMatrix element
                this->elemental_jacobian_matrix()(element, i + j * BasisFunction::nodes_per_element) =
                    basis_functions[i]->integrateGradientWithBasis(*basis_functions[j]);
            }
        }

        // cleanup
        for (BasisFunction*& basis : basis_functions) {
            delete basis;
        }
    }

    // upload to device
    this->elemental_jacobian_matrix().copyToDevice(stream);
}

// forward solving
template <
    class BasisFunction,
    class NumericSolver
>
const fastEIT::Matrix<fastEIT::dtype::real>& fastEIT::ForwardSolver<BasisFunction, NumericSolver>::solve(
    const Matrix<dtype::real>& gamma, dtype::size steps, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("ForwardSolver::solve: handle == NULL");
    }

    // update system matrix
    this->model().update(gamma, handle, stream);

    // solve for ground mode
    this->numeric_solver().solve(*this->model().system_matrices()[0], *this->excitation()[0],
        steps, true, stream, this->phi()[0]);

    // solve for higher harmonics
    for (dtype::index harmonic = 1; harmonic < this->model().num_harmonics() + 1; ++harmonic) {
        this->numeric_solver().solve(*this->model().system_matrices()[harmonic], *this->excitation()[harmonic],
            steps, false, stream, this->phi()[harmonic]);
    }

    // calc jacobian
    forward::calcJacobian<BasisFunction::nodes_per_element>(gamma, *this->phi()[0],
        this->model().mesh().elements(), this->elemental_jacobian_matrix(),
        this->drive_count(), this->measurment_count(), this->model().sigma_ref(),
        false, stream, &this->jacobian());
    for (dtype::index harmonic = 1; harmonic < this->model().num_harmonics() + 1; ++harmonic) {
        forward::calcJacobian<BasisFunction::nodes_per_element>(gamma, *this->phi()[harmonic],
            this->model().mesh().elements(), this->elemental_jacobian_matrix(),
            this->drive_count(), this->measurment_count(), this->model().sigma_ref(),
            true, stream, &this->jacobian());
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    dtype::real alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltage_calculation().data_rows(),
        this->drive_count(), this->voltage_calculation().data_columns(), &alpha,
        this->voltage_calculation().device_data(), this->voltage_calculation().data_rows(),
        this->phi()[0]->device_data(), this->phi()[0]->data_rows(), &beta,
        this->voltage().device_data(), this->voltage().data_rows());

    // add harmonic voltages
    beta = 1.0f;
    for (dtype::index harmonic = 1; harmonic < this->model().num_harmonics() + 1; ++harmonic) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltage_calculation().data_rows(),
            this->drive_count(), this->voltage_calculation().data_columns(), &alpha,
            this->voltage_calculation().device_data(), this->voltage_calculation().data_rows(),
            this->phi()[harmonic]->device_data(), this->phi()[harmonic]->data_rows(), &beta,
            this->voltage().device_data(), this->voltage().data_rows());
    }

    return this->voltage();
}

// specialisation
template class fastEIT::ForwardSolver<fastEIT::basis::Linear, fastEIT::numeric::SparseConjugate>;
