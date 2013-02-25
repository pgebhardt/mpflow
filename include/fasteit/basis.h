// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_BASIS_H
#define FASTEIT_INCLUDE_BASIS_H

// namespace fastEIT
namespace fastEIT {
    // basis namespace
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
                dtype::index) {
                // init member
                this->nodes_ = nodes;
                for (dtype::real& coefficient : this->coefficients()) {
                    coefficient = 0.0f;
                }
            }

            virtual ~Basis() { }

        // evaluation
        public:
            virtual dtype::real evaluate(std::tuple<dtype::real, dtype::real> point) = 0;

        // geometry definition
        public:
            static const dtype::size nodes_per_edge = template_nodes_per_edge;
            static const dtype::size nodes_per_element = template_nodes_per_element;

        public:
            // accessors
            const std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element>& nodes() const { return this->nodes_; }
            const std::array<dtype::real, nodes_per_element>& coefficients() const { return this->coefficients_; }

            // mutators
            std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element>& nodes() { return this->nodes_; }
            std::array<dtype::real, nodes_per_element>& coefficients() { return this->coefficients_; }

        // member
        private:
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

            // evaluation
            virtual dtype::real evaluate(std::tuple<dtype::real, dtype::real> point);
        };
    }
}

#endif
