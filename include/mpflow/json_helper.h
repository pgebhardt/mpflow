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

#ifndef MPFLOW_INCLUDE_JSON_HELPER_H
#define MPFLOW_INCLUDE_JSON_HELPER_H

#ifdef _JSON_H

namespace mpFlow {
namespace jsonHelper {
	// parse a json encoded numeric value with specializations for
	// complex types encoded as an array of two elements
	template <class dataType>
	dataType parseNumericValue(json_value const& value, dataType const def=dataType(0));
}
}

#endif

#endif