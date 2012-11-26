// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace fastEIT::basis;
using namespace std;

// create basis class
Basis::Basis(dtype::real* x, dtype::real* y) {
    // check input
    if (x == NULL) {
        throw invalid_argument("Basis::Basis: x == NULL");
    }
    if (y == NULL) {
        throw invalid_argument("Basis::Basis: y == NULL");
    }

    // create memory
    this->mPoints = new dtype::real[this->nodesPerElement * 2];
    this->mCoefficients = new dtype::real[this->nodesPerElement];

    // init member
    for (dtype::size i = 0; i < this->nodesPerElement; i++) {
        this->mPoints[i * 2 + 0] = x[i];
        this->mPoints[i * 2 + 1] = y[i];
        this->mCoefficients[i] = 0.0;
    }
}

// delete basis class
Basis::~Basis() {
    // cleanup arrays
    delete [] this->mPoints;
    delete [] this->mCoefficients;
}
