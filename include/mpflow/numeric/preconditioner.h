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

#ifndef MPFLOW_INCLDUE_NUMERIC_PRECONDITIONER_H
#define MPFLOW_INCLDUE_NUMERIC_PRECONDITIONER_H

namespace mpFlow {
namespace numeric {
namespace preconditioner {
    // diagnoal preconditioner
    template <class type>
    void diagonal(std::shared_ptr<Matrix<type> const> const matrix, cudaStream_t const stream,
        std::shared_ptr<SparseMatrix<type>> const result);

    template <class type>
    void diagonal(std::shared_ptr<SparseMatrix<type> const> const matrix, cudaStream_t const stream,
        std::shared_ptr<SparseMatrix<type>> const result);
}
}
}

#endif
