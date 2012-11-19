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
    this->mRadius = radius;
    this->mHeight = height;
    this->mNodeCount = nodeCount;
    this->mElementCount = elementCount;
    this->mBoundaryCount = boundaryCount;
    this->mNodes = nodes;
    this->mElements = elements;
    this->mBoundary = boundary;
}

// delete mesh class
Mesh::~Mesh() {
    // cleanup matrices
    linalgcu_matrix_release(&this->mNodes);
    linalgcu_matrix_release(&this->mElements);
    linalgcu_matrix_release(&this->mBoundary);
}
