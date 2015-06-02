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

#include "mpflow/mpflow.h"

template <
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::Solver(
    std::shared_ptr<equationType> const equation,
    std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
    unsigned const components, unsigned const parallelImages,
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: equation == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: handle == nullptr");
    }

    // create forward solver
    this->forwardSolver = std::make_shared<EIT::ForwardSolver<numericalForwardSolverType, equationType>>(
        equation, source, components, handle, stream);

    // create inverse solver
    this->inverseSolver = std::make_shared<solver::Inverse<dataType, numericalInverseSolverType>>(
        equation->mesh, forwardSolver->jacobian, parallelImages, handle, stream);

    // create matrices
    this->referenceDistribution = std::make_shared<numeric::Matrix<dataType>>(
        equation->mesh->elements.rows(), parallelImages, stream,
        equationType::logarithmic ? dataType(0) : equation->referenceValue);
    this->materialDistribution = std::make_shared<numeric::Matrix<dataType>>(
        equation->mesh->elements.rows(), parallelImages, stream,
        equationType::logarithmic ? dataType(0) : equation->referenceValue);
    this->delta = std::make_shared<numeric::Matrix<dataType>>(
        equation->mesh->elements.rows(), parallelImages, stream);
        
    for (unsigned image = 0; image < parallelImages; ++image) {
        this->measurement.push_back(std::make_shared<numeric::Matrix<dataType>>(
            source->measurementPattern->cols, source->drivePattern->cols, stream, 0.0, false));
        this->calculation.push_back(std::make_shared<numeric::Matrix<dataType>>(
            source->measurementPattern->cols, source->drivePattern->cols, stream, 0.0, false));
    }
}

template <class dataType>
dataType parseReferenceValue(json_value const& config) {
    if (config.type == json_double) {
        return config.u.dbl;
    }
    else {
        return 1.0;
    }
}

template <>
thrust::complex<double> parseReferenceValue(json_value const& config) {
    if (config.type == json_array) {
        return thrust::complex<double>(config[0], config[1]);
    }
    else if (config.type == json_double) {
        return thrust::complex<double>(config.u.dbl);
    }
    else {
        return thrust::complex<double>(1.0);
    }
}

template <
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
std::shared_ptr<mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType,
    equationType>> mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType,
    equationType>::fromConfig(json_value const& config, cublasHandle_t const handle,
    cudaStream_t const stream, std::string const path,
    std::shared_ptr<numeric::IrregularMesh const> const externalMesh) {
    // check input
    if (handle == nullptr) {
        return nullptr;
    }

    // extract model config
    auto const modelConfig = config["model"];
    if (modelConfig.type == json_none) {
        return nullptr;
    }

    // extract solver config
    auto const solverConfig = config["solver"];
    if (solverConfig.type == json_none) {
        return nullptr;
    }

    // load electrodes from config
    auto const electrodes = FEM::BoundaryDescriptor::fromConfig(modelConfig["electrodes"],
        modelConfig["mesh"]["radius"].u.dbl);
    if (electrodes == nullptr) {
        return nullptr;
    }

    // load source from config
    auto const source = FEM::SourceDescriptor<dataType>::fromConfig(modelConfig["source"],
        electrodes, stream);
    if (source == nullptr) {
        return nullptr;
    }

    // read out reference value and distribution
    auto const referenceValue = parseReferenceValue<dataType>(modelConfig["material"]);
    auto const referenceDistribution = [=](json_value const& config) {
        if (config.type == json_string) {
            return numeric::Matrix<dataType>::loadtxt(
                str::format("%s/%s")(path, std::string(config)), stream);
        }
        else {
            return std::shared_ptr<numeric::Matrix<dataType>>(nullptr);
        }
    }(modelConfig["material"]);

    // load mesh from config
    auto const mesh = externalMesh != nullptr ? externalMesh :
        numeric::IrregularMesh::fromConfig(modelConfig["mesh"],
        electrodes, stream, path);
    if (mesh == nullptr) {
        return nullptr;
    }

    // initialize equation
    auto equation = std::make_shared<equationType>(mesh, source->electrodes, referenceValue,
        stream);

    // extract parallel images count
    int const parallelImages = std::max(1, (int)solverConfig["parallelImages"].u.integer);

    // create inverse solver and forward model
    auto solver = std::make_shared<EIT::Solver<numericalForwardSolverType,
        numericalInverseSolverType, equationType>>(equation, source,
        std::max(1, (int)modelConfig["componentsCount"].u.integer),
        parallelImages, handle, stream);

    // override reference Distribution, if applicable
    if (referenceDistribution != nullptr) {
        solver->referenceDistribution = referenceDistribution;
    }
    solver->preSolve(handle, stream);

    // clear jacobian matrix for not needed elements
    if (modelConfig["mesh"]["noSensitivity"].type != json_none) {
        solver->forwardSolver->jacobian->elementwiseMultiply(solver->forwardSolver->jacobian,
            numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(modelConfig["mesh"]["noSensitivity"])), stream),
            stream);
        solver->inverseSolver->updateJacobian(solver->forwardSolver->jacobian,
            handle, stream);
    }

    // extract regularization parameter
    double const regularizationFactor = solverConfig["regularizationFactor"].u.dbl;
    auto const regularizationType = [](json_value const& config) {
        typedef solver::Inverse<dataType, numericalInverseSolverType> inverseSolverType;
        
        if (std::string(config) == "diagonal") {
            return inverseSolverType::RegularizationType::diagonal;
        }
        else if (std::string(config) == "totalVariational") {
            return inverseSolverType::RegularizationType::totalVariational;
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
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
void mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::preSolve(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::pre_solve: handle == nullptr");
    }
    
    // forward solving a few steps
    auto const initialValue = this->forwardSolver->solve(this->referenceDistribution,
        handle, stream);

    // calc initial system matrix
    this->inverseSolver->updateJacobian(this->forwardSolver->jacobian, handle, stream);

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
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::solveDifferential(
    cublasHandle_t const handle, cudaStream_t const stream, unsigned const maxIterations,
    unsigned* const iterations) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solve_differential: handle == nullptr");
    }

    // solve
    unsigned const steps = this->inverseSolver->solve(this->calculation, this->measurement,
        maxIterations, handle, stream, this->delta);

    if (iterations != nullptr) {
        *iterations = steps;
    }

    return this->delta;
}

// solve absolute
template <
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::solveAbsolute(
    unsigned const iterations, cublasHandle_t const handle, cudaStream_t const stream) {
    // only execute method, when parallelImages == 1
    if (this->measurement.size() != 1) {
        throw std::runtime_error(
            "mpFlow::EIT::Solver::solveAbsolute: parallelImages != 1");
    }

    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solveAbsolute: handle == nullptr");
    }

    // do newton iterations
    for (unsigned step = 0; step < iterations; ++step) {
        // solve for new jacobian and reference data
        this->forwardSolver->solve(this->materialDistribution, handle, stream);
        this->inverseSolver->updateJacobian(this->forwardSolver->jacobian, handle, stream);
    
        // solve inverse
        this->inverseSolver->solve({ this->forwardSolver->result },
            this->measurement, 0, handle, stream, this->delta);
    
        // add to result
        this->materialDistribution->add(this->delta, stream);
    }
    
    return this->materialDistribution;
}

// specialisation
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;

template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
        
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;

template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
