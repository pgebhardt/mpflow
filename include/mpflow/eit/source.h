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

#ifndef MPFLOW_INCLDUE_EIT_SOURCE_H
#define MPFLOW_INCLDUE_EIT_SOURCE_H

namespace mpFlow {
namespace EIT {
    // source base class
    class Source {
    public:
        // constructor
        Source(std::string type, const std::vector<dtype::real>& values,
            std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
            std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern,
            std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern,
            cudaStream_t stream);
        Source(std::string type, dtype::real value,
            std::shared_ptr<FEM::BoundaryDescriptor> electrodes,
            std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern,
            std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern,
            cudaStream_t stream);
        virtual ~Source() { }

        void updateExcitation(std::shared_ptr<numeric::Matrix<dtype::real>> excitation,
            cublasHandle_t handle, cudaStream_t stream);

        // member
        std::string type;
        std::shared_ptr<FEM::BoundaryDescriptor> electrodes;
        std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern;
        std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern;
        std::shared_ptr<numeric::Matrix<dtype::real>> pattern;
        std::vector<dtype::real> values;
    };

    // source types
    namespace source {
        static std::string const CurrentSourceType = "current";
        static std::string const VoltageSourceType = "voltage";
    }
}
}

#endif
