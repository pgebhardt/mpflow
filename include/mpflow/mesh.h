// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MESH_H
#define FASTEIT_INCLUDE_MESH_H

// namespace fastEIT
namespace fastEIT {
    // mesh class definition
    class Mesh {
    public:
        // constructor
        Mesh(std::shared_ptr<Matrix<dtype::real>> nodes, std::shared_ptr<Matrix<dtype::index>> elements,
            std::shared_ptr<Matrix<dtype::index>> boundary, dtype::real radius, dtype::real height);

        // helper methods
        std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>>
            elementNodes(dtype::index element);
        std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>>
            boundaryNodes(dtype::index element);

        // accessors
        std::shared_ptr<Matrix<dtype::real>> nodes() { return this->nodes_; }
        std::shared_ptr<Matrix<dtype::index>> elements() { return this->elements_; }
        std::shared_ptr<Matrix<dtype::index>> boundary() { return this->boundary_; }
        dtype::real radius() { return this->radius_; }
        dtype::real height() { return this->height_; }

    private:
        // member
        std::shared_ptr<Matrix<dtype::real>> nodes_;
        std::shared_ptr<Matrix<dtype::index>> elements_;
        std::shared_ptr<Matrix<dtype::index>> boundary_;
        dtype::real radius_;
        dtype::real height_;
    };

    // mesh helper
    namespace mesh {
        // create mesh for quadratic basis function
        std::shared_ptr<fastEIT::Mesh> quadraticBasis(std::shared_ptr<Matrix<dtype::real>> nodes,
            std::shared_ptr<Matrix<dtype::index>> elements,
            std::shared_ptr<Matrix<dtype::index>> boundary,
            dtype::real radius, dtype::real height, cudaStream_t stream);

        // quadratic mesh from linear
        std::tuple<
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>> quadraticMeshFromLinear(
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> nodes_old,
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> elements_old,
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> boundary_old,
            cudaStream_t stream);
    }
}

#endif
