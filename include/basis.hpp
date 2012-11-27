// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_HPP
#define FASTEIT_BASIS_HPP

// basis namespace
namespace basis {
    // abstract basis class
    template <
        int templateNodesPerEdge,
        int templateNodesPerElement
    >
    class Basis {
    // constructor and destructor
    protected:
        Basis(dtype::real* x, dtype::real* y) {
            // check input
            if (x == NULL) {
                throw std::invalid_argument("Basis::Basis: x == NULL");
            }
            if (y == NULL) {
                throw std::invalid_argument("Basis::Basis: y == NULL");
            }

            // create memory
            this->mPoints = new dtype::real[this->nodesPerElement * 2];
            this->mCoefficients = new dtype::real[this->nodesPerElement];

            // init member
            for (dtype::size i = 0; i < this->nodesPerElement; i++) {
                this->mPoints[i * 2 + 0] = x[i];
                this->mPoints[i * 2 + 1] = y[i];
                this->mCoefficients[i] = 0.0;
            }
        }

        virtual ~Basis() {
            // cleanup arrays
            delete [] this->mPoints;
            delete [] this->mCoefficients;
        }

    // operator
    public:
        virtual dtype::real operator()(dtype::real x, dtype::real y) = 0;

    // geometry definition
    public:
        static const dtype::size nodesPerEdge = templateNodesPerEdge;
        static const dtype::size nodesPerElement = templateNodesPerElement;

    // access methods
    public:
        const dtype::real* point(dtype::index id) const {
            assert(id < nodesPerElement);
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
    class Linear : public Basis<2, 3> {
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
