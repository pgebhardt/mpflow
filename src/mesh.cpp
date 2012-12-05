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

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/basis.h"
#include "../include/mesh.h"

// create mesh class
template <
    class BasisFunction
>
fastEIT::Mesh<BasisFunction>::Mesh(const Matrix<dtype::real>& nodes,
    const Matrix<dtype::index>& elements, const Matrix<dtype::index>& boundary,
    dtype::real radius, dtype::real height, cudaStream_t stream)
    : radius_(radius), height_(height), nodes_(NULL), elements_(NULL),
        boundary_(NULL) {
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

    // create matrices
    this->nodes_ = new Matrix<dtype::real>(nodes.rows(), nodes.columns(), stream);
    this->elements_ = new Matrix<dtype::index>(elements.rows(), elements.columns(), stream);
    this->boundary_ = new Matrix<dtype::index>(boundary.rows(), boundary.columns(), stream);

    // copy matrices
    this->nodes_->copy(nodes, stream);
    this->elements_->copy(elements, stream);
    this->boundary_->copy(boundary, stream);

    // copy to host
    this->nodes_->copyToHost(stream);
    this->elements_->copyToHost(stream);
    this->boundary_->copyToHost(stream);
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
        indices[node] = this->elements()(element, node);
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
        nodes[node] = std::make_tuple(this->nodes()(indices[node], 0),
            this->nodes()(indices[node], 1));
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
        indices[node] = this->boundary()(bound, node);
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
        nodes[node] = std::make_tuple(this->nodes()(indices[node], 0),
            this->nodes()(indices[node], 1));
    }

    return nodes;
}

template class fastEIT::Mesh<fastEIT::basis::Linear>;
