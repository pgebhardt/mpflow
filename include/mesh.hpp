// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_MESH_HPP
#define FASTEIT_MESH_HPP

// mesh class definition
class Mesh {
// constructor and destructor
public:
    Mesh(Matrix<dtype::real>* nodes, Matrix<dtype::index>* elements,
        Matrix<dtype::index>* boundary, dtype::real radius,
        dtype::real height);
    virtual ~Mesh();

// access methods
public:
    Matrix<dtype::real>* nodes() const { return this->mNodes; }
    Matrix<dtype::index>* elements() const { return this->mElements; }
    Matrix<dtype::index>* boundary() const { return this->mBoundary; }
    dtype::real radius() const { return this->mRadius; }
    dtype::real height() const { return this->mHeight; }

// member
private:
    Matrix<dtype::real>* mNodes;
    Matrix<dtype::index>* mElements;
    Matrix<dtype::index>* mBoundary;
    dtype::real mRadius;
    dtype::real mHeight;
};

#endif
