// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create electrodes class
Electrodes::Electrodes(linalgcuSize_t count, linalgcuMatrixData_t width, linalgcuMatrixData_t height,
    linalgcuMatrixData_t meshRadius) {
    // check input
    if (count == 0) {
        throw invalid_argument("count");
    }
    if (width <= 0.0f) {
        throw invalid_argument("width");
    }
    if (height <= 0.0f) {
        throw invalid_argument("height");
    }
    if (meshRadius <= 0.0f) {
        throw invalid_argument("meshRadius");
    }

    // init member
    this->mCount = count;
    this->mElectrodesStart = NULL;
    this->mElectrodesEnd = NULL;
    this->mWidth = width;
    this->mHeight = height;

    // create electrode vectors
    this->mElectrodesStart = new linalgcuMatrixData_t[this->mCount * 2];
    this->mElectrodesEnd = new linalgcuMatrixData_t[this->mCount * 2];

    // fill electrode vectors
    linalgcuMatrixData_t angle = 0.0f;
    linalgcuMatrixData_t delta_angle = M_PI / (linalgcuMatrixData_t)this->mCount;
    for (linalgcuSize_t i = 0; i < this->mCount; i++) {
        // calc start angle
        angle = (linalgcuMatrixData_t)i * 2.0f * delta_angle;

        // calc start coordinates
        this->mElectrodesStart[i * 2 + 0] = meshRadius * cos(angle);
        this->mElectrodesStart[i * 2 + 1] = meshRadius * sin(angle);

        // calc end angle
        angle += this->mWidth / meshRadius;

        // calc end coordinates
        this->mElectrodesEnd[i * 2 + 0] = meshRadius * cos(angle);
        this->mElectrodesEnd[i * 2 + 1] = meshRadius * sin(angle);
    }
}

// delete electrodes class
Electrodes::~Electrodes() {
    // free electrode vectors
    if (this->mElectrodesStart != NULL) {
        delete [] this->mElectrodesStart;
    }
    if (this->mElectrodesEnd != NULL) {
        delete [] this->mElectrodesEnd;
    }
}
