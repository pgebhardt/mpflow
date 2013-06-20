// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_CONSTANTS_H
#define MPFLOW_INCLUDE_CONSTANTS_H

//namespace mpFlow
namespace mpFlow {
namespace matrix {
    // matrix block size
    const mpFlow::dtype::size block_size = 16;
}

namespace sparseMatrix {
    // sparse matrix block size
    const mpFlow::dtype::size block_size = 32;
}
}

#endif
