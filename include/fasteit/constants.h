// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_CONSTANTS_H
#define FASTEIT_INCLUDE_CONSTANTS_H

//namespace fastEIT
namespace fastEIT {
    // matrix block size
    namespace matrix {
        const fastEIT::dtype::size block_size = 16;
    }

    // sparse matrix block size
    namespace sparseMatrix {
        const fastEIT::dtype::size block_size = 32;
    }
}

#endif
