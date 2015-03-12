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
    class SourceDescriptor {
    public:
        enum Type {
            Open,
            Fixed
        };

        // constructor
        SourceDescriptor(Type type, const std::vector<dtype::real>& values,
            std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
            std::shared_ptr<numeric::Matrix<dtype::integral>> drivePattern,
            std::shared_ptr<numeric::Matrix<dtype::integral>> measurementPattern,
            cudaStream_t stream);
        SourceDescriptor(Type type, dtype::real value,
            std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
            std::shared_ptr<numeric::Matrix<dtype::integral>> drivePattern,
            std::shared_ptr<numeric::Matrix<dtype::integral>> measurementPattern,
            cudaStream_t stream);
        virtual ~SourceDescriptor() { }

        // member
        Type type;
        std::shared_ptr<FEM::BoundaryDescriptor> electrodes;
        std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern;
        std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern;
        std::shared_ptr<numeric::Matrix<dtype::real>> pattern;
        std::vector<dtype::real> values;
    };
}
}

#endif
