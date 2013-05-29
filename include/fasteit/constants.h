// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_CONSTANTS_H
#define FASTEIT_INCLUDE_CONSTANTS_H

//namespace fastEIT
namespace fastEIT {
namespace matrix {
    // matrix block size
    const fastEIT::dtype::size block_size = 16;
}

namespace sparseMatrix {
    // sparse matrix block size
    const fastEIT::dtype::size block_size = 32;
}
}

#endif
