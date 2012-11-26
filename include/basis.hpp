// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_HPP
#define FASTEIT_BASIS_HPP

// basis namespace
namespace basis {
    // abstract basis class
    class Basis {
    // constructor and destructor
    protected:
        Basis(dtype::real* x, dtype::real* y);
        virtual ~Basis();

    // operator
    public:
        virtual dtype::real operator()(dtype::real x, dtype::real y) = 0;

    // geometry definition
    public:
        static const dtype::size nodesPerEdge = 2;
        static const dtype::size nodesPerElement = 3;

    // access methods
    public:
        const dtype::real* point(dtype::index id) const {
            assert(id <nodesPerElement);
            return &this->mPoints[id * 2];
        }
        dtype::real coefficient(dtype::index id) const {
            assert(id < nodesPerElement);
            return this->mCoefficients[id];
        }

    protected:
        dtype::real& setCoefficient(dtype::index id) {
            assert(id < nodesPerElement);
            return this->mCoefficients[id];
        }

    // member
    private:
        dtype::real* mPoints;
        dtype::real* mCoefficients;
    };

    // linear basis class definition
    class Linear : public Basis {
    // constructor
    public:
        Linear(dtype::real* x, dtype::real* y);

    // mathematical evaluation of basis
    public:
        virtual dtype::real integrateWithBasis(Linear& other);
        virtual dtype::real integrateGradientWithBasis(Linear& other);
        static dtype::real integrateBoundaryEdge(dtype::real* x, dtype::real* y,
            dtype::real* start, dtype::real* end);

    // operator
    public:
        virtual dtype::real operator()(dtype::real x, dtype::real y);
    };
}

#endif
