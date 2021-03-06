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
    class dataType,
    bool logarithmic
>
mpFlow::models::Constant<dataType, logarithmic>::Constant(
    std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<FEM::Sources<dataType> const> const sources,
    std::shared_ptr<numeric::Matrix<dataType>> const jacobian, dataType const referenceValue,
    cudaStream_t const stream)
    : mesh(mesh), sources(sources), jacobian(jacobian), referenceValue(referenceValue) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::models::Constant::Constant: mesh == nullptr");
    }
    if (sources == nullptr) {
        throw std::invalid_argument("mpFlow::models::Constant::Constant: sources == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::models::Constant::Constant: jacobian == nullptr");
    }
    
    this->result = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->cols, this->sources->drivePattern->cols, stream);
}

template <
    class dataType,
    bool logarithmic
>
std::shared_ptr<mpFlow::models::Constant<dataType>>
    mpFlow::models::Constant<dataType, logarithmic>::fromConfig(
    json_value const& config, cublasHandle_t const, cudaStream_t const stream,
    std::string const path, std::shared_ptr<numeric::IrregularMesh const> const externalMesh) {
    // load mesh from config
    auto const mesh = externalMesh != nullptr ? externalMesh :
        numeric::IrregularMesh::fromConfig(config["mesh"], config["ports"], stream, path);

    // load ports descriptor from config
    auto const ports = FEM::Ports::fromConfig(
        config["ports"], mesh, stream, path);

    // load sources from config
    auto const sources = FEM::Sources<dataType>::fromConfig(
        config["source"], ports, stream);

    // load jacobian from config
    auto const jacobian = numeric::Matrix<dataType>::loadtxt(
        str::format("%s/%s")(path, std::string(config["jacobian"])), stream);
        
    // read out reference value
    auto const referenceValue = config["material"].type == json_object ?
        jsonHelper::parseNumericValue<dataType>(config["material"]["referenceValue"], 1.0) :
        jsonHelper::parseNumericValue<dataType>(config["material"], 1.0);
    
    // create forward model
    return std::make_shared<Constant<dataType>>(mesh, sources, jacobian, referenceValue, stream);
}

// forward solving
template <
    class dataType,
    bool logarithmic
>
std::shared_ptr<mpFlow::numeric::Matrix<dataType> const>
    mpFlow::models::Constant<dataType, logarithmic>::solve(
    std::shared_ptr<numeric::Matrix<dataType> const> const,
    cublasHandle_t const, cudaStream_t const, unsigned* const steps) {
    if (steps != nullptr) {
        *steps = 0;
    }
    
    return this->result;
}

// specialisation
template class mpFlow::models::Constant<float, false>;
template class mpFlow::models::Constant<float, true>;
template class mpFlow::models::Constant<double, false>;
template class mpFlow::models::Constant<double, true>;
template class mpFlow::models::Constant<thrust::complex<float>, false>;
template class mpFlow::models::Constant<thrust::complex<float>, true>;
template class mpFlow::models::Constant<thrust::complex<double>, false>;
template class mpFlow::models::Constant<thrust::complex<double>, true>;