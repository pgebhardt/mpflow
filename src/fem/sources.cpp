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
mpFlow::FEM::Sources<dataType>::Sources(Type const type,
    std::vector<dataType> const& values, std::shared_ptr<FEM::Ports const> const ports,
    std::shared_ptr<numeric::Matrix<int> const> const drivePattern,
    std::shared_ptr<numeric::Matrix<int> const> const measurementPattern,
    cudaStream_t const stream)
    : type(type), values(values), ports(ports) {
    // check input
    if (ports == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::Sources::Sources: ports == nullptr");
    }
    if (drivePattern == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::Sources::Sources: drivePattern == nullptr");
    }
    if (measurementPattern == nullptr) {
        throw std::invalid_argument(
            "mpFlow::FEM::Sources::Sources: measurementPattern == nullptr");
    }
    if (values.size() != drivePattern->cols) {
        throw std::invalid_argument(
            "mpFlow::FEM::Sources::Sources: invalid size of values vector");
    }

    // create matrices
    this->drivePattern = std::make_shared<numeric::Matrix<dataType>>(drivePattern->rows,
        drivePattern->cols, stream);
    this->measurementPattern = std::make_shared<numeric::Matrix<dataType>>(measurementPattern->rows,
        measurementPattern->cols, stream);
    this->pattern = std::make_shared<numeric::Matrix<dataType>>(this->ports->count,
        this->drivePattern->cols + this->measurementPattern->cols, stream);

    // fill pattern matrix with drive pattern
    for (unsigned row = 0; row < this->ports->count; ++row)
    for (unsigned col = 0; col < this->drivePattern->cols; ++col) {
        (*this->drivePattern)(row, col) = (*drivePattern)(row, col);
        (*this->pattern)(row, col) = this->values[col] * (*this->drivePattern)(row, col);
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (unsigned row = 0; row < this->ports->count; ++row)
    for (unsigned col = 0; col < this->measurementPattern->cols; ++col) {
        (*this->measurementPattern)(row, col) = (*measurementPattern)(row, col);
        (*this->pattern)(row, col + this->drivePattern->cols) =
            (*this->measurementPattern)(row, col);
    }
    this->drivePattern->copyToDevice(stream);
    this->measurementPattern->copyToDevice(stream);
    this->pattern->copyToDevice(stream);
}

template <
    class dataType
>
mpFlow::FEM::Sources<dataType>::Sources(Type const type, dataType const value,
    std::shared_ptr<FEM::Ports const> const ports,
    std::shared_ptr<numeric::Matrix<int> const> const drivePattern,
    std::shared_ptr<numeric::Matrix<int> const> const measurementPattern,
    cudaStream_t const stream)
    : Sources<dataType>(type, std::vector<dataType>(drivePattern->cols, value),
        ports, drivePattern, measurementPattern, stream) {
}

template <
    class dataType
>
std::shared_ptr<mpFlow::FEM::Sources<dataType>> mpFlow::FEM::Sources<dataType>::fromConfig(
    json_value const& config, std::shared_ptr<Ports const> const ports,
    cudaStream_t const stream) {
    // function to parse pattern config
    auto const parsePatternConfig = [=](json_value const& config)
        -> std::shared_ptr<numeric::Matrix<int>> {
        if (config.type != json_none) {
            return numeric::Matrix<int>::fromJsonArray(config, stream);
        }
        else {
            return numeric::Matrix<int>::eye(ports->count, stream);
        }
    };
    
    // load excitation and measurement pattern from config or assume standard pattern, if not given
    auto const drivePattern = parsePatternConfig(config["drivePattern"]);
    auto const measurementPattern = parsePatternConfig(config["measurementPattern"]);
    
    // read out currents
    std::vector<dataType> excitation(drivePattern->cols);
    if (config["value"].type == json_array) {
        for (unsigned i = 0; i < drivePattern->cols; ++i) {
            excitation[i] = config["value"][i].u.dbl;
        }
    }
    else {
        excitation = std::vector<dataType>(drivePattern->cols,
            config["value"].type != json_none ? config["value"].u.dbl : dataType(1));
    }

    // create source descriptor
    auto const sourceType = std::string(config["type"]) == "voltage" ?
        mpFlow::FEM::Sources<dataType>::Type::Fixed :
        mpFlow::FEM::Sources<dataType>::Type::Open;
    auto source = std::make_shared<mpFlow::FEM::Sources<dataType>>(sourceType,
        excitation, ports, drivePattern, measurementPattern, stream);

    return source;

}

// specialisation
template class mpFlow::FEM::Sources<float>;
template class mpFlow::FEM::Sources<double>;
template class mpFlow::FEM::Sources<thrust::complex<float>>;
template class mpFlow::FEM::Sources<thrust::complex<double>>;
