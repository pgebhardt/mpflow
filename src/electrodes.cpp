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
    this->mElectrodesStart = new dtype::real[this->count() * 2];
    this->mElectrodesEnd = new dtype::real[this->count() * 2];

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real deltaAngle = M_PI / (dtype::real)this->mCount;
    for (dtype::index i = 0; i < this->mCount; i++) {
        // calc start angle
        angle = (dtype::real)i * 2.0f * deltaAngle;

        // calc start coordinates
        this->electrodesStart(i)[0] = meshRadius * cos(angle);
        this->electrodesStart(i)[1] = meshRadius * sin(angle);

        // calc end angle
        angle += this->width() / meshRadius;

        // calc end coordinates
        this->electrodesEnd(i)[0] = meshRadius * cos(angle);
        this->electrodesEnd(i)[1] = meshRadius * sin(angle);
    }
}

// delete electrodes class
Electrodes::~Electrodes() {
    // free electrode vectors
    delete [] this->mElectrodesStart;
    delete [] this->mElectrodesEnd;
}
