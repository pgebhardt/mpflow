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
    Matrix<dtype::index>* boundary, dtype::real radius,
    dtype::real height)
    : mRadius(radius), mHeight(height), mNodes(nodes), mElements(elements),
        mBoundary(boundary) {
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
