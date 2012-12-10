// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <assert.h>
#include <stdexcept>
#include <tuple>
#include <array>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/basis.h"
#include "../include/mesh.h"

// create mesh class
template <
    class BasisFunction
>
fastEIT::Mesh<BasisFunction>::Mesh(std::shared_ptr<Matrix<dtype::real>> nodes,
    std::shared_ptr<Matrix<dtype::index>> elements, std::shared_ptr<Matrix<dtype::index>> boundary,
    dtype::real radius, dtype::real height, cudaStream_t stream)
    : radius_(radius), height_(height), nodes_(nodes), elements_(elements),
        boundary_(boundary) {
    // check input
    if (radius <= 0.0f) {
        throw std::invalid_argument("radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw std::invalid_argument("height <= 0.0");
    }
    if (elements->columns() != BasisFunction::nodes_per_element) {
        throw std::invalid_argument("elements.count() != BasisFunction::nodes_per_element");
    }
    if (boundary->columns() != BasisFunction::nodes_per_edge) {
        throw std::invalid_argument("boundary.count() != BasisFunction::nodes_per_edge");
    }
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
        indices[node] = (*this->elements())(element, node);
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
        nodes[node] = std::make_tuple((*this->nodes())(indices[node], 0),
            (*this->nodes())(indices[node], 1));
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
        indices[node] = (*this->boundary())(bound, node);
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
        nodes[node] = std::make_tuple((*this->nodes())(indices[node], 0),
            (*this->nodes())(indices[node], 1));
    }

    return nodes;
}

template class fastEIT::Mesh<fastEIT::basis::Linear>;
