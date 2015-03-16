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
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"

template <
    class dataType
>
mpFlow::FEM::SourceDescriptor<dataType>::SourceDescriptor(Type type,
    const std::vector<dataType>& values, std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
    std::shared_ptr<numeric::Matrix<int>> drivePattern,
    std::shared_ptr<numeric::Matrix<int>> measurementPattern,
    cudaStream_t stream)
    : type(type), electrodes(electrodes), values(values) {
    // check input
    if (electrodes == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::SourceDescriptor::SourceDescriptor: electrodes == nullptr");
    }
    if (drivePattern == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::SourceDescriptor::SourceDescriptor: drivePattern == nullptr");
    }
    if (measurementPattern == nullptr) {
        throw std::invalid_argument(
            "mpFlow::FEM::SourceDescriptor::SourceDescriptor: measurementPattern == nullptr");
    }
    if (values.size() != drivePattern->cols) {
        throw std::invalid_argument(
            "mpFlow::FEM::SourceDescriptor::SourceDescriptor: invalid size of values vector");
    }

    // create matrices
    this->drivePattern = std::make_shared<numeric::Matrix<dataType>>(drivePattern->rows,
        drivePattern->cols, stream);
    this->measurementPattern = std::make_shared<numeric::Matrix<dataType>>(measurementPattern->rows,
        measurementPattern->cols, stream);
    this->pattern = std::make_shared<numeric::Matrix<dataType>>(this->electrodes->count,
        this->drivePattern->cols + this->measurementPattern->cols, stream);

    // fill pattern matrix with drive pattern
    for (unsigned row = 0; row < this->electrodes->count; ++row)
    for (unsigned col = 0; col < this->drivePattern->cols; ++col) {
        (*this->drivePattern)(row, col) = (*drivePattern)(row, col);
        (*this->pattern)(row, col) = this->values[col] * (*this->drivePattern)(row, col);
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (unsigned row = 0; row < this->electrodes->count; ++row)
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
mpFlow::FEM::SourceDescriptor<dataType>::SourceDescriptor(Type type, dataType value,
    std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
    std::shared_ptr<numeric::Matrix<int>> drivePattern,
    std::shared_ptr<numeric::Matrix<int>> measurementPattern,
    cudaStream_t stream)
    : SourceDescriptor<dataType>(type, std::vector<dataType>(drivePattern->cols, value),
        electrodes, drivePattern, measurementPattern, stream) {
}

// specialisation
template class mpFlow::FEM::SourceDescriptor<float>;
template class mpFlow::FEM::SourceDescriptor<double>;
template class mpFlow::FEM::SourceDescriptor<thrust::complex<float>>;
template class mpFlow::FEM::SourceDescriptor<thrust::complex<double>>;
