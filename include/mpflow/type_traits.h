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

#ifndef MPFLOW_INCLUDE_TYPE_TRAITS_H
#define MPFLOW_INCLUDE_TYPE_TRAITS_H

namespace mpFlow {
namespace typeTraits {
    // extract the numerical type of the data type, specialized for
    // thrust::complex and std::complex
    template <typename type_>
    struct extractNumericalType { typedef type_ type; };

    template <typename type_>
    struct extractNumericalType<thrust::complex<type_> > { typedef type_ type; };

    template <typename type_>
    struct extractNumericalType<std::complex<type_> > { typedef type_ type; };

    // convert thrust::complex to std::complex and vice versa
    template <typename type_>
    struct convertComplexType { typedef type_ type; };

    template <typename type_>
    struct convertComplexType<thrust::complex<type_> > { typedef std::complex<type_> type; };

    template <typename type_>
    struct convertComplexType<std::complex<type_> > { typedef thrust::complex<type_> type; };
}
}

#endif
