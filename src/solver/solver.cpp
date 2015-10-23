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

#include "json.h"
#include "mpflow/mpflow.h"

template <
    class forwardModelType,
    template <class> class numericalInverseSolverType
>
mpFlow::solver::Solver<forwardModelType, numericalInverseSolverType>::Solver(
    std::shared_ptr<forwardModelType> const forwardModel, unsigned const parallelImages,
    cublasHandle_t const handle, cudaStream_t const stream)
    : forwardModel(forwardModel) {
    // check input
    if (forwardModel == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Solver::Solver: forwardModel == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Solver::Solver: handle == nullptr");
    }

    // create inverse solver
    this->inverseSolver = std::make_shared<solver::Inverse<dataType, numericalInverseSolverType>>(
        this->forwardModel->mesh, this->forwardModel->jacobian, forwardModel->referenceValue,
        parallelImages, handle, stream);

    // create matrices
    for (unsigned image = 0; image < parallelImages; ++image) {
        this->measurement.push_back(std::make_shared<numeric::Matrix<dataType>>(
            this->forwardModel->sources->measurementPattern->cols,
            this->forwardModel->sources->drivePattern->cols, stream, 0.0, false));
        this->calculation.push_back(std::make_shared<numeric::Matrix<dataType>>(
            this->forwardModel->sources->measurementPattern->cols,
            this->forwardModel->sources->drivePattern->cols, stream, 0.0, false));
    }

    dataType const initialValue = forwardModelType::logarithmic ? dataType(0) :
        dataType(1);
    this->referenceDistribution = std::make_shared<numeric::Matrix<dataType>>(
        this->forwardModel->mesh->elements.rows(), parallelImages, stream,
        initialValue);
    this->materialDistribution = std::make_shared<numeric::Matrix<dataType>>(
        this->forwardModel->mesh->elements.rows(), parallelImages, stream,
        initialValue);
}

template <
    class forwardModelType,
    template <class> class numericalInverseSolverType
>
std::shared_ptr<mpFlow::solver::Solver<forwardModelType, numericalInverseSolverType>>
    mpFlow::solver::Solver<forwardModelType, numericalInverseSolverType>::fromConfig(
    json_value const& config, cublasHandle_t const handle, cudaStream_t const stream,
    std::string const path, std::shared_ptr<numeric::IrregularMesh const> const externalMesh) {
    // extract model and solver configurations
    auto const modelConfig = config["model"];
    auto const solverConfig = config["solver"];

    // load forward model from config
    auto const forwardModel = forwardModelType::fromConfig(modelConfig, handle, stream, path, externalMesh);

    // read out reference distribution
    auto const referenceDistribution = [=](json_value const& config) {
        if (config.type == json_string) {
            return numeric::Matrix<dataType>::loadtxt(
                str::format("%s/%s")(path, std::string(config)), stream);
        }
        else if ((config.type == json_object) && (config["distribution"].type == json_string)) {
            return numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(config["distribution"])), stream);
        }
        else {
            return std::shared_ptr<numeric::Matrix<dataType>>(nullptr);
        }
    }(modelConfig["material"]);

    // extract parallel images count
    auto const parallelImages = std::max(1, (int)solverConfig["parallelImages"].u.integer);

    // create inverse solver and forward model
    auto const solver = std::make_shared<solver::Solver<forwardModelType, numericalInverseSolverType>>(
        forwardModel, parallelImages, handle, stream);

    // override reference Distribution, if applicable
    if (referenceDistribution != nullptr) {
        solver->referenceDistribution = referenceDistribution;
    }
    solver->preSolve(handle, stream);

    // clear jacobian matrix for not needed elements
    if (modelConfig["noSensitivity"].type == json_string) {
        forwardModel->jacobian->elementwiseMultiply(forwardModel->jacobian,
            numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(modelConfig["noSensitivity"])), stream),
            stream);
        solver->inverseSolver->updateJacobian(forwardModel->jacobian, handle, stream);
    }

    // extract regularization parameter
    auto const regularizationFactor = jsonHelper::parseNumericValue<dataType>(solverConfig["regularizationFactor"]);
    auto const regularizationType = [](json_value const& config) {
        typedef solver::Inverse<dataType, numericalInverseSolverType> inverseSolverType;

        if (std::string(config) == "noser") {
            return inverseSolverType::RegularizationType::noser;
        }
        else if (std::string(config) == "laplacian") {
            return inverseSolverType::RegularizationType::laplacian;
        }
        else {
            return inverseSolverType::RegularizationType::identity;
        }
    }(solverConfig["regularizationType"]);

    solver->inverseSolver->setRegularizationParameter(regularizationFactor, regularizationType,
        handle, stream);

    return solver;
}

// pre solve for accurate initial jacobian
template <
    class forwardModelType,
    template <class> class numericalInverseSolverType
>
void mpFlow::solver::Solver<forwardModelType, numericalInverseSolverType>::preSolve(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Solver::preSolve: handle == nullptr");
    }

    // forward solving a few steps
    auto const initialValue = this->forwardModel->solve(this->referenceDistribution,
        handle, stream);

    // calc initial system matrix
    this->inverseSolver->updateJacobian(this->forwardModel->jacobian, handle, stream);

    // set measurement and calculation to initial value of forward model
    for (auto level : this->measurement) {
        level->copy(initialValue, stream);
    }
    for (auto level : this->calculation) {
        level->copy(initialValue, stream);
    }
}

// solve differential
template <
    class forwardModelType,
    template <class> class numericalInverseSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename forwardModelType::dataType> const>
    mpFlow::solver::Solver<forwardModelType, numericalInverseSolverType>::solveDifferential(
    cublasHandle_t const handle, cudaStream_t const stream, unsigned const maxIterations,
    unsigned* const iterations) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::solver::Solver::solveDifferential: handle == nullptr");
    }

    // solve
    auto const result = this->inverseSolver->solve(this->calculation, this->measurement,
        handle, stream, maxIterations, iterations);

    this->materialDistribution->copy(this->referenceDistribution, stream);
    this->materialDistribution->add(result, stream);

    return this->materialDistribution;
}

// solve absolute
template <
    class forwardModelType,
    template <class> class numericalInverseSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename forwardModelType::dataType> const>
    mpFlow::solver::Solver<forwardModelType, numericalInverseSolverType>::solveAbsolute(
    unsigned const iterations, cublasHandle_t const handle, cudaStream_t const stream) {
    // only execute method, when parallelImages == 1
    if (this->measurement.size() != 1) {
        throw std::runtime_error(
            "mpFlow::solver::Solver::solveAbsolute: parallelImages != 1");
    }

    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::solver::Solver::solveAbsolute: handle == nullptr");
    }

    // do newton iterations
    for (unsigned step = 0; step < iterations; ++step) {
        // solve for new jacobian and reference data
        this->forwardModel->solve(this->materialDistribution, handle, stream);
        this->inverseSolver->updateJacobian(this->forwardModel->jacobian, handle, stream);

        // solve inverse
        auto const result = this->inverseSolver->solve({ this->forwardModel->result },
            this->measurement, handle, stream);

        // add to result
        this->materialDistribution->add(result, stream);
    }

    return this->materialDistribution;
}

// specialisation
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;

template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;

template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;

template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>, mpFlow::numeric::BiCGSTAB>;
    
template class mpFlow::solver::Solver<mpFlow::models::Constant<float>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<float>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<double>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<double>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<thrust::complex<float>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<thrust::complex<float>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<thrust::complex<double>>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<mpFlow::models::Constant<thrust::complex<double>>, mpFlow::numeric::BiCGSTAB>;

template class mpFlow::solver::Solver<mpFlow::models::MWI<mpFlow::numeric::CPUSolver,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Edge, false>>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Solver<mpFlow::models::MWI<mpFlow::numeric::CPUSolver,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Edge, false>>, mpFlow::numeric::BiCGSTAB>;    
