// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"
#include "../include/mesh.hpp"

// create mesh class
fastEIT::Mesh::Mesh(Matrix<dtype::real>& nodes, Matrix<dtype::index>& elements,
    Matrix<dtype::index>& boundary, dtype::real radius, dtype::real height)
    : radius_(radius), height_(height), nodes_(&nodes), elements_(&elements),
        boundary_(&boundary) {
    // check input
    if (radius <= 0.0f) {
        throw std::invalid_argument("radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw std::invalid_argument("height <= 0.0");
    }
}

// delete mesh class
fastEIT::Mesh::~Mesh() {
    // cleanup matrices
    delete this->nodes_;
    delete this->elements_;
    delete this->boundary_;
}
