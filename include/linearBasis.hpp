// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_HPP
#define FASTEIT_BASIS_HPP

// basis class definition
class LinearBasis {
// constructer and destructor
public:
    LinearBasis(dtype::real* x, dtype::real* y);
    virtual ~LinearBasis();

// mathematical evaluation of basis
public:
    dtype::real evaluate(dtype::real x, dtype::real y);
    dtype::real integrate_with_basis(LinearBasis& other);
    dtype::real integrate_gradient_with_basis(LinearBasis& other);
    static dtype::real integrate_boundary_edge(dtype::real* x,
        dtype::real* y, dtype::real* start, dtype::real* end);

// operator
public:
    dtype::real operator()(dtype::real x, dtype::real y);

// geometry definition
public:
    static const dtype::size nodesPerEdge = 2;
    static const dtype::size nodesPerElement = 3;

// access methods
public:
    inline dtype::real* point(dtype::size id) const {
        assert(id < LinearBasis::nodesPerElement);
        return &this->mPoints[id * 2]; 
    }
    inline dtype::real coefficient(dtype::size id) const {
        assert(id < LinearBasis::nodesPerElement);
        return this->mCoefficients[id];
    }

// member
private:
    dtype::real* mPoints;
    dtype::real* mCoefficients;
};

#endif
