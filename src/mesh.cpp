// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <tuple>
#include <array>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"
#include "../include/basis.hpp"
#include "../include/mesh.hpp"

// create mesh class
template <
    class BasisFunction
>
fastEIT::Mesh<BasisFunction>::Mesh(Matrix<dtype::real>& nodes, Matrix<dtype::index>& elements,
    Matrix<dtype::index>& boundary, dtype::real radius, dtype::real height)
    : radius_(radius), height_(height), nodes_(&nodes), elements_(&elements),
        boundary_(&boundary) {
    // check input
    if (radius <= 0.0f) {
        throw std::invalid_argument("radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw std::invalid_argument("height <= 0.0");
    }
    if (elements.columns() != BasisFunction::nodes_per_element) {
        throw std::invalid_argument("elements.count() != BasisFunction::nodes_per_element");
    }
    if (boundary.columns() != BasisFunction::nodes_per_edge) {
        throw std::invalid_argument("boundary.count() != BasisFunction::nodes_per_edge");
    }
}

// delete mesh class
template <
    class BasisFunction
>
fastEIT::Mesh<BasisFunction>::~Mesh() {
    // cleanup matrices
    delete this->nodes_;
    delete this->elements_;
    delete this->boundary_;
}

template <
    class BasisFunction
>
std::array<fastEIT::dtype::index, BasisFunction::nodes_per_element> fastEIT::Mesh<BasisFunction>::elementIndices(
    dtype::index element) const {
    // needed variables
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;

    // get nodes
    for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
        // get index
        indices[node] = this->elements().get(element, node);
    }

    return indices;
}

template <
    class BasisFunction
>
std::array<std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>, BasisFunction::nodes_per_element>
    fastEIT::Mesh<BasisFunction>::elementNodes(dtype::index element) const {
    // nodes array
    std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_element> nodes;

    // get indices
    auto indices = this->elementIndices(element);

    // get nodes
    for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
        // get coordinates
        nodes[node] = std::make_tuple(this->nodes().get(indices[node], 0),
            this->nodes().get(indices[node], 1));
    }

    return nodes;
}

template <
    class BasisFunction
>
std::array<fastEIT::dtype::index, BasisFunction::nodes_per_edge> fastEIT::Mesh<BasisFunction>::boundaryIndices(
    dtype::index bound) const {
    // needed variables
    std::array<dtype::index, BasisFunction::nodes_per_edge> indices;

    // get nodes
    for (dtype::index node = 0; node < BasisFunction::nodes_per_edge; ++node) {
        // get index
        indices[node] = this->boundary().get(bound, node);
    }

    return indices;
}

template <
    class BasisFunction
>
std::array<std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>, BasisFunction::nodes_per_edge>
    fastEIT::Mesh<BasisFunction>::boundaryNodes(dtype::index bound) const {
    // nodes array
    std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_edge> nodes;

    // get indices
    auto indices = this->boundaryIndices(bound);

    // get nodes
    for (dtype::index node = 0; node < BasisFunction::nodes_per_edge; ++node) {
        // get coordinates
        nodes[node] = std::make_tuple(this->nodes().get(indices[node], 0),
            this->nodes().get(indices[node], 1));
    }

    return nodes;
}

template class fastEIT::Mesh<fastEIT::basis::Linear>;
