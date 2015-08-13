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

// parse a json encoded numeric value with specializations for
// complex types encoded as an array of two elements
template <class dataType>
dataType mpFlow::jsonHelper::parseNumericValue(json_value const& value,
	dataType const def) {
	if (value.type == json_double) {
		return value.u.dbl;
	}
	else {
		return def;
	}		
}

template float mpFlow::jsonHelper::parseNumericValue<float>(json_value const&, float const);
template double mpFlow::jsonHelper::parseNumericValue<double>(json_value const&, double const);

template <>
thrust::complex<float> mpFlow::jsonHelper::parseNumericValue(json_value const& value,
	thrust::complex<float> const def) {
	if ((value.type == json_array) && (value.u.array.length >= 2)) {
		return thrust::complex<float>(value[0].u.dbl, value[1].u.dbl);
	}
	else if (value.type == json_double) {
		return thrust::complex<float>(value.u.dbl);
	}
	else {
		return def;
	}
}

template <>
thrust::complex<double> mpFlow::jsonHelper::parseNumericValue(json_value const& value,
	thrust::complex<double> const def) {
	if ((value.type == json_array) && (value.u.array.length >= 2)) {
		return thrust::complex<double>(value[0].u.dbl, value[1].u.dbl);
	}
	else if (value.type == json_double) {
		return thrust::complex<double>(value.u.dbl);
	}
	else {
		return def;
	}
}