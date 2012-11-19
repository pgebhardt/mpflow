// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create mesh class
Mesh::Mesh(linalgcuMatrix_t nodes, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
    linalgcuSize_t nodeCount, linalgcuSize_t elementCount, linalgcuSize_t boundaryCount,
    linalgcuMatrixData_t radius, linalgcuMatrixData_t height) {
    // check input
    if (nodes == NULL) {
        throw invalid_argument("nodes");
    }
    if (elements == NULL) {
        throw invalid_argument("elements");
    }
    if (boundary == NULL) {
        throw invalid_argument("boundary");
    }
    if (nodeCount > nodes->rows) {
        throw invalid_argument("nodeCount");
    }
    if (elementCount > elements->rows) {
        throw invalid_argument("elementCount");
    }
    if (boundaryCount > boundary->rows) {
        throw invalid_argument("boundaryCount");
    }
    if (radius <= 0.0f) {
        throw invalid_argument("radius");
    }
    if (height <= 0.0f) {
        throw invalid_argument("height");
    }

    // init member
    this->radius() = radius;
    this->height() = height;
    this->nodeCount() = nodeCount;
    this->elementCount() = elementCount;
    this->boundaryCount() = boundaryCount;
    this->nodes() = nodes;
    this->elements() = elements;
    this->boundary() = boundary;
}

// delete mesh class
Mesh::~Mesh() {
    // cleanup matrices
    linalgcu_matrix_release(&this->nodes());
    linalgcu_matrix_release(&this->elements());
    linalgcu_matrix_release(&this->boundary());
}
