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

#ifndef MPFLOW_INCLDUE_FEM_SOURCES_H
#define MPFLOW_INCLDUE_FEM_SOURCES_H

namespace mpFlow {
namespace FEM {
    template <
        class dataType
    >
    class Sources {
    public:
        enum Type {
            Open,
            Fixed
        };

        // constructor
        Sources(Type const type, std::vector<dataType> const& values,
            std::shared_ptr<FEM::Ports const> const ports,
            std::shared_ptr<numeric::Matrix<int> const> const drivePattern,
            std::shared_ptr<numeric::Matrix<int> const> const measurementPattern,
            cudaStream_t const stream);
        Sources(Type const type, dataType const value,
            std::shared_ptr<FEM::Ports const> const ports,
            std::shared_ptr<numeric::Matrix<int> const> const drivePattern,
            std::shared_ptr<numeric::Matrix<int> const> const measurementPattern,
            cudaStream_t const stream);
        virtual ~Sources() { }

        // factories
#ifdef _JSON_H        
        static std::shared_ptr<Sources<dataType>> fromConfig(json_value const& config,
            std::shared_ptr<Ports const> const ports,
            cudaStream_t const stream);
#endif

        // member
        Type const type;
        std::vector<dataType> const values;
        std::shared_ptr<FEM::Ports const> const ports;
        std::shared_ptr<numeric::Matrix<dataType>> drivePattern;
        std::shared_ptr<numeric::Matrix<dataType>> measurementPattern;
        std::shared_ptr<numeric::Matrix<dataType>> pattern;
    };
}
}

#endif
