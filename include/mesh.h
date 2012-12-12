// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MESH_H
#define FASTEIT_INCLUDE_MESH_H

// namespace fastEIT
namespace fastEIT {
    // mesh class definition
    template <
        class BasisFunction
    >
    class Mesh {
    public:
        // constructor
        Mesh(std::shared_ptr<Matrix<dtype::real>> nodes, std::shared_ptr<Matrix<dtype::index>> elements,
            std::shared_ptr<Matrix<dtype::index>> boundary, dtype::real radius, dtype::real height);

        // helper methods
        std::array<dtype::index, BasisFunction::nodes_per_element> elementIndices(dtype::index element) const;
        std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_element>
            elementNodes(dtype::index element) const;

        std::array<dtype::index, BasisFunction::nodes_per_edge> boundaryIndices(dtype::index bound) const;
        std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_edge>
            boundaryNodes(dtype::index bound) const;

        // accessors
        const std::shared_ptr<Matrix<dtype::real>> nodes() const { return this->nodes_; }
        const std::shared_ptr<Matrix<dtype::index>> elements() const { return this->elements_; }
        const std::shared_ptr<Matrix<dtype::index>> boundary() const { return this->boundary_; }
        dtype::real radius() const { return this->radius_; }
        dtype::real height() const { return this->height_; }

        // mutators
        std::shared_ptr<Matrix<dtype::real>> nodes() { return this->nodes_; }
        std::shared_ptr<Matrix<dtype::index>> elements() { return this->elements_; }
        std::shared_ptr<Matrix<dtype::index>> boundary() { return this->boundary_; }
        dtype::real& radius() { return this->radius_; }
        dtype::real& height() { return this->height_; }

    protected:
        // member
        std::shared_ptr<Matrix<dtype::real>> nodes_;
        std::shared_ptr<Matrix<dtype::index>> elements_;
        std::shared_ptr<Matrix<dtype::index>> boundary_;
        dtype::real radius_;
        dtype::real height_;
    };

    // mesh helper
    namespace mesh {
        // quadratic mesh from linear
        std::tuple<
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>> quadraticMeshFromLinear(
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> nodes_old,
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> elements_old,
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> boundary_old);
    }
}

#endif
