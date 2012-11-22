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
    Mesh(linalgcuMatrix_t nodes, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
        dtype::size nodeCount, dtype::size elementCount, dtype::size boundaryCount,
        dtype::real radius, dtype::real height);
    virtual ~Mesh();

// access methods
public:
    linalgcuMatrix_t nodes() const { return this->mNodes; }
    linalgcuMatrix_t elements() const { return this->mElements; }
    linalgcuMatrix_t boundary() const { return this->mBoundary; }
    dtype::size nodeCount() const { return this->mNodeCount; }
    dtype::size elementCount() const { return this->mElementCount; }
    dtype::size boundaryCount() const { return this->mBoundaryCount; }
    dtype::real radius() const { return this->mRadius; }
    dtype::real height() const { return this->mHeight; }

// member
private:
    linalgcuMatrix_t mNodes;
    linalgcuMatrix_t mElements;
    linalgcuMatrix_t mBoundary;
    dtype::size mNodeCount;
    dtype::size mElementCount;
    dtype::size mBoundaryCount;
    dtype::real mRadius;
    dtype::real mHeight;
};

#endif
