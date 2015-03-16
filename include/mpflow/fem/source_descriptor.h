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

#ifndef MPFLOW_INCLDUE_FEM_SOURCE_DESCRIPTOR_H
#define MPFLOW_INCLDUE_FEM_SOURCE_DESCRIPTOR_H

namespace mpFlow {
namespace FEM {
    template <
        class dataType
    >
    class SourceDescriptor {
    public:
        enum Type {
            Open,
            Fixed
        };

        // constructor
        SourceDescriptor(Type type, const std::vector<dataType>& values,
            std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
            std::shared_ptr<numeric::Matrix<int>> drivePattern,
            std::shared_ptr<numeric::Matrix<int>> measurementPattern,
            cudaStream_t stream);
        SourceDescriptor(Type type, dataType value,
            std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
            std::shared_ptr<numeric::Matrix<int>> drivePattern,
            std::shared_ptr<numeric::Matrix<int>> measurementPattern,
            cudaStream_t stream);
        virtual ~SourceDescriptor() { }

        // member
        Type type;
        std::shared_ptr<FEM::BoundaryDescriptor> electrodes;
        std::shared_ptr<numeric::Matrix<dataType>> drivePattern;
        std::shared_ptr<numeric::Matrix<dataType>> measurementPattern;
        std::shared_ptr<numeric::Matrix<dataType>> pattern;
        std::vector<dataType> values;
    };
}
}

#endif
