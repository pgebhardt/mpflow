// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_HPP
#define FASTEIT_BASIS_HPP

// namespace fastEIT
namespace fastEIT {
    // basis class definition
    class Basis {
    // constructer and destructor
    public:
        Basis(linalgcuMatrixData_t* x, linalgcuMatrixData_t* y);
        virtual ~Basis();

    // mathematical evaluation of basis
    public:
        linalgcuMatrixData_t evaluate(linalgcuMatrixData_t x, linalgcuMatrixData_t y);
        linalgcuMatrixData_t integrate_with_basis(Basis& other);
        linalgcuMatrixData_t integrate_gradient_with_basis(Basis& other);
        static linalgcuMatrixData_t integrate_boundary_edge(linalgcuMatrixData_t* x,
            linalgcuMatrixData_t* y, linalgcuMatrixData_t* start, linalgcuMatrixData_t* end);

    // operator
    public:
        linalgcuMatrixData_t operator()(linalgcuMatrixData_t x, linalgcuMatrixData_t y);

    // access methods
    public:
        inline linalgcuMatrixData_t* point(linalgcuSize_t id) const {
            assert(id < 3);
            return &this->mPoints[id * 2]; 
        }
        inline linalgcuMatrixData_t coefficient(linalgcuSize_t id) const {
            assert(id < 3);
            return this->mCoefficients[id];
        }

    // geometry definition
    public:
        static const linalgcuSize_t nodesPerEdge = 2;
        static const linalgcuSize_t nodesPerElement = 3;

    // member
    private:
        linalgcuMatrixData_t* mPoints;
        linalgcuMatrixData_t* mCoefficients;
    };
}

#endif
