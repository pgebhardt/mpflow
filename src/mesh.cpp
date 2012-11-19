// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

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
        throw invalid_argument("nodes == NULL");
    }
    if (elements == NULL) {
        throw invalid_argument("elements == NULL");
    }
    if (boundary == NULL) {
        throw invalid_argument("boundary == NULL");
    }
    if (nodeCount > nodes->rows) {
        throw invalid_argument("nodeCount > nodes->rows");
    }
    if (elementCount > elements->rows) {
        throw invalid_argument("elementCount > elements->rows");
    }
    if (boundaryCount > boundary->rows) {
        throw invalid_argument("boundaryCount > boundary->rows");
    }
    if (radius <= 0.0f) {
        throw invalid_argument("radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw invalid_argument("height <= 0.0");
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
