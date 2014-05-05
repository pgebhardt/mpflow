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

mpFlow::EIT::Source::Source(std::string type, const std::vector<dtype::real>& values,
    std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
    std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern,
    std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern,
    cudaStream_t stream)
    : type(type), electrodes(electrodes), drivePattern(drivePattern),
        measurementPattern(measurementPattern), values(values) {
    // check input
    if (electrodes == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Source::Source: electrodes == nullptr");
    }
    if (drivePattern == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Source::Source: drivePattern == nullptr");
    }
    if (measurementPattern == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Source::Source: measurementPattern == nullptr");
    }
    if (values.size() != this->drivePattern->columns()) {
        throw std::invalid_argument(
            "mpFlow::EIT::Source::Source: invalid size of values vector");
    }

    // create matrices
    this->pattern = std::make_shared<numeric::Matrix<dtype::real>>(this->electrodes->count,
        this->drivePattern->columns() + this->measurementPattern->columns(), stream);

    // fill pattern matrix with drive pattern
    for (dtype::index column = 0; column < this->drivePattern->columns(); ++column)
    for (dtype::index row = 0; row < this->electrodes->count; ++row) {
        (*this->pattern)(row, column) = (*this->drivePattern)(row, column);
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::index column = 0; column < this->measurementPattern->columns(); ++column)
    for (dtype::index row = 0; row < this->electrodes->count; ++row) {
        (*this->pattern)(row, column + this->drivePattern->columns()) =
            (*this->measurementPattern)(row, column);
    }
    this->pattern->copyToDevice(stream);
}

mpFlow::EIT::Source::Source(std::string type, dtype::real value,
    std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
    std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern,
    std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern,
    cudaStream_t stream)
    : Source(type, std::vector<dtype::real>(drivePattern->columns(), value),
        electrodes, drivePattern, measurementPattern, stream) {
}

void mpFlow::EIT::Source::updateExcitation(std::shared_ptr<numeric::Matrix<dtype::real>> excitation,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (excitation == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Source::updateExcitation: excitation == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Source::updateExcitation: handle == nullptr");
    }

    cublasSetStream(handle, stream);
    for (dtype::index i = 0; i < this->pattern->columns(); ++i) {
        if (cublasScopy(handle, this->pattern->rows(),
            this->pattern->device_data() + i * this->pattern->data_rows(), 1,
            excitation->device_data() + i * excitation->data_rows() +
            excitation->rows(), 1) != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::EIT::Source::updateExcitation: copy pattern to excitation");
        }
    }

    for (dtype::index i = 0; i < this->drivePattern->columns(); ++i) {
        if (cublasSscal(handle, this->pattern->rows(), &this->values[i],
            excitation->device_data() + i * excitation->data_rows() +
            excitation->rows(), 1) != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::EIT::source::Current::updateExcitation: apply value to pattern");
        }
    }
}
