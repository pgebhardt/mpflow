// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_HPP
#define FASTEIT_BASIS_HPP

// basis namespace
namespace Basis {
    // basis class definition
    class Linear {
    // constructer and destructor
    public:
        Linear(fastEIT::dtype::real* x, fastEIT::dtype::real* y);
        virtual ~Linear();

    // mathematical evaluation of basis
    public:
        fastEIT::dtype::real integrateWithBasis(fastEIT::Basis::Linear& other);
        dtype::real integrateGradientWithBasis(fastEIT::Basis::Linear& other);
        static dtype::real integrateBoundaryEdge(fastEIT::dtype::real* x,
            fastEIT::dtype::real* y, fastEIT::dtype::real* start, fastEIT::dtype::real* end);

    // operator
    public:
        fastEIT::dtype::real operator()(fastEIT::dtype::real x, fastEIT::dtype::real y);

    // geometry definition
    public:
        static const fastEIT::dtype::size nodesPerEdge = 2;
        static const fastEIT::dtype::size nodesPerElement = 3;

    // access methods
    public:
        inline fastEIT::dtype::real* point(fastEIT::dtype::size id) const {
            assert(id < fastEIT::Basis::Linear::nodesPerElement);
            return &this->mPoints[id * 2]; 
        }
        inline fastEIT::dtype::real coefficient(fastEIT::dtype::size id) const {
            assert(id < fastEIT::Basis::Linear::nodesPerElement);
            return this->mCoefficients[id];
        }

    // member
    private:
        fastEIT::dtype::real* mPoints;
        fastEIT::dtype::real* mCoefficients;
    };
}

#endif
