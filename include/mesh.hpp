// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_MESH_HPP
#define FASTEIT_MESH_HPP

// namespace fastEIT
namespace fastEIT {
    // mesh class definition
    class Mesh {
    // constructor and destructor
    public:
        Mesh(linalgcuMatrix_t nodes, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
            linalgcuSize_t nodeCount, linalgcuSize_t elementCount, linalgcuSize_t boundaryCount,
            linalgcuMatrixData_t radius, linalgcuMatrixData_t height);
        virtual ~Mesh();

    // access methods
    public:
        linalgcuMatrix_t nodes() const { return this->mNodes; }
        linalgcuMatrix_t elements() const { return this->mElements; }
        linalgcuMatrix_t boundary() const { return this->mBoundary; }
        linalgcuSize_t nodeCount() const { return this->mNodeCount; }
        linalgcuSize_t elementCount() const { return this->mElementCount; }
        linalgcuSize_t boundaryCount() const { return this->mBoundaryCount; }
        linalgcuMatrixData_t radius() const { return this->mRadius; }
        linalgcuMatrixData_t height() const { return this->mHeight; }

    // member
    private:
        linalgcuMatrix_t mNodes;
        linalgcuMatrix_t mElements;
        linalgcuMatrix_t mBoundary;
        linalgcuSize_t mNodeCount;
        linalgcuSize_t mElementCount;
        linalgcuSize_t mBoundaryCount;
        linalgcuMatrixData_t mRadius;
        linalgcuMatrixData_t mHeight;
    };
}

#endif
