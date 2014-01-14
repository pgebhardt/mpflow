// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_FEM_BASIS_H
#define MPFLOW_INCLUDE_FEM_BASIS_H

// namespace mpFlow::FEM::basis
namespace mpFlow {
namespace FEM {
namespace basis {
    // abstract basis class
    template <
        int template_nodes_per_edge,
        int template_nodes_per_element
    >
    class Basis {
    // constructor and destructor
    protected:
        Basis(std::array<std::tuple<dtype::real, dtype::real>, template_nodes_per_element> nodes,
            dtype::index)
            : nodes_(nodes) {
            for (auto& coefficient : this->coefficients()) {
                coefficient = 0.0f;
            }
        }

        virtual ~Basis() { }

    public:
        // evaluation
        virtual dtype::real evaluate(std::tuple<dtype::real, dtype::real> point) = 0;

        // geometry definition
        static const dtype::size nodes_per_edge = template_nodes_per_edge;
        static const dtype::size nodes_per_element = template_nodes_per_element;

        // accessors
        std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element>& nodes() { return this->nodes_; }
        std::array<dtype::real, nodes_per_element>& coefficients() { return this->coefficients_; }

    private:
        // member
        std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes_;
        std::array<dtype::real, nodes_per_element> coefficients_;
    };

    // linear basis class definition
    class Linear : public Basis<2, 3> {
    public:
        // constructor
        Linear(std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
            dtype::index one);

        // mathematical evaluation of basis
        virtual dtype::real integrateWithBasis(const std::shared_ptr<Linear> other);
        virtual dtype::real integrateGradientWithBasis(const std::shared_ptr<Linear> other);
        static dtype::real integrateBoundaryEdge(
            std::array<dtype::real, nodes_per_edge> nodes, dtype::index one,
            dtype::real start, dtype::real end);
        static dtype::real integrateBoundaryEdgeWithOther(
            std::array<dtype::real, nodes_per_edge> nodes, dtype::index self,
            dtype::index other, dtype::real start, dtype::real end);

        // evaluation
        virtual dtype::real evaluate(std::tuple<dtype::real, dtype::real> point);
    };

    // quadratic basis class definition
    class Quadratic : public Basis<3, 6> {
    public:
        // constructor
        Quadratic(std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
            dtype::index one);

        // mathematical evaluation of basis
        virtual dtype::real integrateWithBasis(const std::shared_ptr<Quadratic> other);
        virtual dtype::real integrateGradientWithBasis(const std::shared_ptr<Quadratic> other);
        static dtype::real integrateBoundaryEdge(
            std::array<dtype::real, nodes_per_edge> nodes, dtype::index one,
            dtype::real start, dtype::real end);
        static dtype::real integrateBoundaryEdgeWithOther(
            std::array<dtype::real, nodes_per_edge> nodes, dtype::index self,
            dtype::index other, dtype::real start, dtype::real end);

        // evaluation
        virtual dtype::real evaluate(std::tuple<dtype::real, dtype::real> point);
    };
}
}
}

#endif
