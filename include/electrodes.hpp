// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_ELECTRODES_HPP
#define FASTEIT_ELECTRODES_HPP

// Electrodes class definition
class Electrodes {
// constructer and destructor
public:
    Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real meshRadius);
    virtual ~Electrodes();

// access methods
public:
    dtype::size count() const { return this->mCount; }
    dtype::real* electrodesStart() const { return this->mElectrodesStart; }
    dtype::real* electrodesEnd() const { return this->mElectrodesEnd; }
    dtype::real width() const { return this->mWidth; }
    dtype::real height() const { return this->mHeight; }

// member
private:
    dtype::size mCount;
    dtype::real* mElectrodesStart;
    dtype::real* mElectrodesEnd;
    dtype::real mWidth;
    dtype::real mHeight;
};

#endif
