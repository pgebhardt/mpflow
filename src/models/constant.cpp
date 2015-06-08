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
    class dataType
>
mpFlow::models::Constant<dataType>::Constant(std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
    std::shared_ptr<numeric::Matrix<dataType>> const jacobian, dataType const referenceValue,
    cudaStream_t const stream)
    : mesh(mesh), source(source), jacobian(jacobian), referenceValue(referenceValue) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::models::Constant::Constant: mesh == nullptr");
    }
    if (source == nullptr) {
        throw std::invalid_argument("mpFlow::models::Constant::Constant: source == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::models::Constant::Constant: jacobian == nullptr");
    }
    
    this->result = std::make_shared<numeric::Matrix<dataType>>(
        this->source->measurementPattern->cols, this->source->drivePattern->cols, stream);
}

template <class dataType>
static dataType parseReferenceValue(json_value const& config) {
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
    class dataType
>
std::shared_ptr<mpFlow::models::Constant<dataType>> mpFlow::models::Constant<dataType>::fromConfig(
    json_value const& config, cublasHandle_t const, cudaStream_t const stream,
    std::string const path, std::shared_ptr<numeric::IrregularMesh const> const externalMesh) {
    // load boundary descriptor from config
    auto const boundaryDescriptor = FEM::BoundaryDescriptor::fromConfig(config["boundary"],
        config["mesh"]["radius"].u.dbl);
    if (boundaryDescriptor == nullptr) {
        return nullptr;
    }

    // load source from config
    auto const source = FEM::SourceDescriptor<dataType>::fromConfig(
        config["source"], boundaryDescriptor, stream);
    if (source == nullptr) {
        return nullptr;
    }

    // load mesh from config
    auto const mesh = externalMesh != nullptr ? externalMesh :
        numeric::IrregularMesh::fromConfig(config["mesh"], boundaryDescriptor, stream, path);
    if (mesh == nullptr) {
        return nullptr;
    }

    // load jacobian from config
    auto jacobian = numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(config["jacobian"])), stream);
        
    // read out reference value
    auto const referenceValue = parseReferenceValue<dataType>(config["material"]);
    
    // create forward model
    return std::make_shared<Constant<dataType>>(mesh, source, jacobian, referenceValue, stream);
}

// forward solving
template <
    class dataType
>
std::shared_ptr<mpFlow::numeric::Matrix<dataType> const>
    mpFlow::models::Constant<dataType>::solve(
    std::shared_ptr<numeric::Matrix<dataType> const> const,
    cublasHandle_t const, cudaStream_t const, unsigned* const steps) {
    if (steps != nullptr) {
        *steps = 0;
    }
    
    return this->result;
}

// specialisation
template class mpFlow::models::Constant<float>;
template class mpFlow::models::Constant<double>;
template class mpFlow::models::Constant<thrust::complex<float>>;
template class mpFlow::models::Constant<thrust::complex<double>>;