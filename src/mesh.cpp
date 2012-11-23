// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create mesh class
Mesh::Mesh(Matrix<dtype::real>* nodes, Matrix<dtype::index>* elements,
    Matrix<dtype::index>* boundary, dtype::size nodeCount,
    dtype::size elementCount, dtype::size boundaryCount, dtype::real radius,
    dtype::real height)
    : mRadius(radius), mHeight(height), mNodeCount(nodeCount), mElementCount(elementCount),
        mBoundaryCount(boundaryCount), mNodes(nodes), mElements(elements), mBoundary(boundary) {
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
    if (nodeCount > nodes->dataRows()) {
        throw invalid_argument("nodeCount > nodes->rows");
    }
    if (elementCount > elements->dataRows()) {
        throw invalid_argument("elementCount > elements->rows");
    }
    if (boundaryCount > boundary->dataRows()) {
        throw invalid_argument("boundaryCount > boundary->rows");
    }
    if (radius <= 0.0f) {
        throw invalid_argument("radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw invalid_argument("height <= 0.0");
    }
}

// delete mesh class
Mesh::~Mesh() {
    // cleanup matrices
    delete this->mNodes;
    delete this->mElements;
    delete this->mBoundary;
}
