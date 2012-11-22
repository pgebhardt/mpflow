// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_HPP
#define FASTEIT_BASIS_HPP

// namespace fastEIT
namespace fastEIT {
    // basis class definition
    class LinearBasis {
    // constructer and destructor
    public:
        LinearBasis(linalgcuMatrixData_t* x, linalgcuMatrixData_t* y);
        virtual ~LinearBasis();

    // mathematical evaluation of basis
    public:
        linalgcuMatrixData_t evaluate(linalgcuMatrixData_t x, linalgcuMatrixData_t y);
        linalgcuMatrixData_t integrate_with_basis(LinearBasis& other);
        linalgcuMatrixData_t integrate_gradient_with_basis(LinearBasis& other);
        static linalgcuMatrixData_t integrate_boundary_edge(linalgcuMatrixData_t* x,
            linalgcuMatrixData_t* y, linalgcuMatrixData_t* start, linalgcuMatrixData_t* end);

    // operator
    public:
        linalgcuMatrixData_t operator()(linalgcuMatrixData_t x, linalgcuMatrixData_t y);

    // geometry definition
    public:
        static const linalgcuSize_t nodesPerEdge = 2;
        static const linalgcuSize_t nodesPerElement = 3;

    // access methods
    public:
        inline linalgcuMatrixData_t* point(linalgcuSize_t id) const {
            assert(id < LinearBasis::nodesPerElement);
            return &this->mPoints[id * 2]; 
        }
        inline linalgcuMatrixData_t coefficient(linalgcuSize_t id) const {
            assert(id < LinearBasis::nodesPerElement);
            return this->mCoefficients[id];
        }


    // member
    private:
        linalgcuMatrixData_t* mPoints;
        linalgcuMatrixData_t* mCoefficients;
    };
}

#endif
