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
#include "../include/basis.h"
#include "../include/mesh.h"
#include "../include/electrodes.h"
#include "../include/conjugate.h"
#include "../include/sparse_conjugate.h"
#include "../include/model.h"
#include "../include/forward.h"
#include "../include/inverse.h"
#include "../include/solver.h"

// create solver
fastEIT::Solver::Solver(Mesh<basis::Linear>* mesh, Electrodes* electrodes,
    const Matrix<dtype::real>& measurment_pattern,
    const Matrix<dtype::real>& drive_pattern, dtype::real sigma_ref,
    dtype::size num_harmonics, dtype::real regularization_factor, cublasHandle_t handle,
    cudaStream_t stream)
    : forward_solver_(NULL), inverse_solver_(NULL), dgamma_(NULL), gamma_(NULL),
        measured_voltage_(NULL), calibration_voltage_(NULL) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Solver::Solver: handle == NULL");
    }

    // create solver
    this->forward_solver_ = new ForwardSolver<basis::Linear, numeric::SparseConjugate>(mesh, electrodes,
        measurment_pattern, drive_pattern, sigma_ref, num_harmonics, handle, stream);

    this->inverse_solver_ = new InverseSolver<numeric::Conjugate>(mesh->elements().rows(),
        measurment_pattern.data_columns() * drive_pattern.data_columns(), regularization_factor, handle, stream);

    // create matrices
    this->dgamma_ = new Matrix<dtype::real>(mesh->elements().rows(), 1, stream);
    this->gamma_ = new Matrix<dtype::real>(mesh->elements().rows(), 1, stream);
    this->measured_voltage_ = new Matrix<dtype::real>(this->forward_solver().measurment_count(),
        this->forward_solver().drive_count(), stream);
    this->calibration_voltage_ = new Matrix<dtype::real>(this->forward_solver().measurment_count(),
        this->forward_solver().drive_count(), stream);
}

// release solver
fastEIT::Solver::~Solver() {
    // cleanup
    delete this->forward_solver_;
    delete this->inverse_solver_;
    delete this->dgamma_;
    delete this->gamma_;
    delete this->measured_voltage_;
    delete this->calibration_voltage_;
}

// pre solve for accurate initial jacobian
void fastEIT::Solver::preSolve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Solver::pre_solve: handle == NULL");
    }

    // forward solving a few steps
    this->forward_solver().solve(this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverse_solver().calcSystemMatrix(this->forward_solver().jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    this->measured_voltage().copy(this->forward_solver().voltage(), stream);
    this->calibration_voltage().copy(this->forward_solver().voltage(), stream);
}

// calibrate
const fastEIT::Matrix<fastEIT::dtype::real>& fastEIT::Solver::calibrate(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Solver::calibrate: handle == NULL");
    }

    // solve forward
    this->forward_solver().solve(this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverse_solver().calcSystemMatrix(this->forward_solver().jacobian(), handle, stream);

    // solve inverse
    this->inverse_solver().solve(this->forward_solver().jacobian(), this->forward_solver().voltage(),
        this->calibration_voltage(), 90, true, handle, stream, &this->dgamma());

    // add to gamma
    this->gamma().add(this->dgamma(), stream);

    return this->gamma();
}

// solving
const fastEIT::Matrix<fastEIT::dtype::real>& fastEIT::Solver::solve(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Solver::solve: handle == NULL");
    }

    // solve
    this->inverse_solver().solve(this->forward_solver().jacobian(), this->calibration_voltage(),
        this->measured_voltage(), 90, false, handle, stream, &this->dgamma());

    return this->dgamma();
}
