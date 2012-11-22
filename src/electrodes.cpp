// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create electrodes class
Electrodes::Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real meshRadius)
    : mCount(count), mElectrodesStart(NULL), mElectrodesEnd(NULL), mWidth(width), mHeight(height) {
    // check input
    if (count == 0) {
        throw invalid_argument("count == 0");
    }
    if (width <= 0.0f) {
        throw invalid_argument("width <= 0.0");
    }
    if (height <= 0.0f) {
        throw invalid_argument("height <= 0.0");
    }
    if (meshRadius <= 0.0f) {
        throw invalid_argument("meshRadius <= 0.0");
    }

    // create electrode vectors
    this->mElectrodesStart = new dtype::real[this->mCount * 2];
    this->mElectrodesEnd = new dtype::real[this->mCount * 2];

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real delta_angle = M_PI / (dtype::real)this->mCount;
    for (dtype::size i = 0; i < this->mCount; i++) {
        // calc start angle
        angle = (dtype::real)i * 2.0f * delta_angle;

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
