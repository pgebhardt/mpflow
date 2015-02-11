// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"

// create mesh class
mpFlow::numeric::IrregularMesh::IrregularMesh(std::shared_ptr<numeric::Matrix<dtype::real>> nodes,
    std::shared_ptr<numeric::Matrix<dtype::index>> elements,
    std::shared_ptr<numeric::Matrix<dtype::index>> boundary, dtype::real radius, dtype::real height)
    : nodes(nodes), elements(elements), boundary(boundary), radius(radius),
        height(height) {
    // check input
    if (nodes == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: nodes == nullptr");
    }
    if (nodes->cols != 2) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: nodes->cols != 2");
    }
    if (elements == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: elements == nullptr");
    }
    if (boundary == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: boundary == nullptr");
    }
    if (radius <= 0.0f) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: height <= 0.0");
    }
}

std::tuple<Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1>,
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>>
    mpFlow::numeric::IrregularMesh::elementNodes(dtype::index element) {
    // initialize output arrays
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1> indices =
        Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1>::Zero(this->elements->cols);
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> coordinates =
        Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>::Zero(this->elements->cols, 2);

    // get node index and coordinate
    for (dtype::index node = 0; node < this->elements->cols; ++node) {
        // get index
        indices(node) = (*this->elements)(element, node);

        // get coordinates
        coordinates(node, 0) = (*this->nodes)(indices(node), 0);
        coordinates(node, 1) = (*this->nodes)(indices(node), 1);
    }

    return std::make_tuple(indices, coordinates);
}

std::tuple<Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1>,
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>>
    mpFlow::numeric::IrregularMesh::boundaryNodes(dtype::index element) {
    // initialize output arrays
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1> indices =
        Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1>::Zero(this->boundary->cols);
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> coordinates =
        Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>::Zero(this->boundary->cols, 2);

    // get node index and coordinate
    for (dtype::index node = 0; node < this->boundary->cols; ++node) {
        // get index
        indices(node) = (*this->boundary)(element, node);

        // get coordinates
        coordinates(node, 0) = (*this->nodes)(indices(node), 0);
        coordinates(node, 1) = (*this->nodes)(indices(node), 1);
    }

    return std::make_tuple(indices, coordinates);
}

// create mesh for quadratic basis function
std::shared_ptr<mpFlow::numeric::IrregularMesh> mpFlow::numeric::irregularMesh::quadraticBasis(
    std::shared_ptr<numeric::Matrix<dtype::real>> nodes,
    std::shared_ptr<numeric::Matrix<dtype::index>> elements, std::shared_ptr<numeric::Matrix<dtype::index>> boundary,
    dtype::real radius, dtype::real height, cudaStream_t stream) {
    // create quadratic grid
    std::tie(nodes, elements, boundary) = quadraticMeshFromLinear(nodes, elements, boundary,
        stream);

    // create mesh
    return std::make_shared<numeric::IrregularMesh>(nodes, elements, boundary, radius, height);
}

// function create quadratic mesh from linear
std::tuple<
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>,
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>>
mpFlow::numeric::irregularMesh::quadraticMeshFromLinear(
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes_old,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements_old,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary_old,
    cudaStream_t stream) {
    // define vectors for calculation
    std::vector<std::array<mpFlow::dtype::index, 2> > already_calc_midpoints(0);
    std::vector<std::array<mpFlow::dtype::real, 2> > new_calc_nodes(0);
    std::vector<std::array<mpFlow::dtype::index, 6> > quadratic_elements_vector(
        elements_old->rows);
    std::vector<std::array<mpFlow::dtype::index, 2> > current_edge(2);
    std::vector<std::array<mpFlow::dtype::real, 2> > quadratic_node_vector(0);
    std::vector<std::array<mpFlow::dtype::real, 2> > node_from_linear(1);
    std::vector<std::array<mpFlow::dtype::index, 3> > quadratic_boundary_vector(0);
    std::vector<std::array<mpFlow::dtype::index, 3> > quadratic_bound(1);
    std::vector<std::array<mpFlow::dtype::index, 2> > linear_bound_inverted(1);
    std::vector<std::array<mpFlow::dtype::index, 2> > linear_bound(1);
    mpFlow::dtype::index new_bound = 0;

    // copy existing elements vom matrix to vector
    for (mpFlow::dtype::index element = 0; element < elements_old->rows; ++element)
    for (mpFlow::dtype::index element_node = 0; element_node < 3; ++element_node) {
        quadratic_elements_vector[element][element_node] = (*elements_old)(element, element_node);
    }

    // calculate new midpoints between existing ones
    for (mpFlow::dtype::index element = 0; element < elements_old->rows; ++element)
    for (mpFlow::dtype::index element_node = 0; element_node < elements_old->cols;
        ++element_node) {
        // get current edge
        current_edge[0][0] = (*elements_old)(element, element_node);
        current_edge[0][1] = (*elements_old)(element,
            (element_node + 1)%elements_old->cols);

        // get current edge inverted
        current_edge[1][0] = (*elements_old)(element,
            (element_node + 1)%elements_old->cols);
        current_edge[1][1] = (*elements_old)(element, element_node);

        //check if midpoint for current adge was calculated before
        auto index = std::find(already_calc_midpoints.begin(), already_calc_midpoints.end(),
            current_edge[0]);
        if(index != already_calc_midpoints.end()) {
            // midpoint already exists, using existing one. check for inverted coords too
            quadratic_elements_vector[element][element_node + 3] = nodes_old->rows +
                std::distance(already_calc_midpoints.begin(), index);
        } else {
            index = std::find(already_calc_midpoints.begin(),
                already_calc_midpoints.end(), current_edge[1]);

            if(index != already_calc_midpoints.end()) {
                quadratic_elements_vector[element][element_node + 3] = nodes_old->rows +
                std::distance(already_calc_midpoints.begin(), index);
            } else {
                // midpoint does not exist. calculate new one
                mpFlow::dtype::real node_x = 0.5 * ((*nodes_old)(current_edge[0][0], 0) +
                    (*nodes_old)(current_edge[0][1],0));
                mpFlow::dtype::real node_y = 0.5 * ((*nodes_old)(current_edge[0][0], 1) +
                    (*nodes_old)(current_edge[0][1],1));

                // create array with x and y coords
                std::array<mpFlow::dtype::real, 2> midpoint = {{node_x, node_y}};

                // append new midpoint to existing nodes
                new_calc_nodes.push_back(midpoint);

                // append current edge to 'already done' list
                already_calc_midpoints.push_back(current_edge[0]);

                // set new node for current element andd adjust node index
                quadratic_elements_vector[element][element_node + 3] = 
                    (new_calc_nodes.size() - 1 + nodes_old->rows);
            }
        }
    }

    // copy nodes from linear mesh and new nodes to one vector
    for (mpFlow::dtype::index node = 0; node < nodes_old->rows; ++node) {
        node_from_linear[0][0] = (*nodes_old)(node, 0);
        node_from_linear[0][1] = (*nodes_old)(node, 1);
        quadratic_node_vector.push_back(node_from_linear[0]);
    }
    for (mpFlow::dtype::index new_node = 0; new_node < new_calc_nodes.size(); ++new_node) {
        quadratic_node_vector.push_back(new_calc_nodes[new_node]);
    }

    // calculate new boundary Matrix
    for(mpFlow::dtype::index bound = 0; bound < boundary_old->rows; ++bound){
        // get current bound
        linear_bound[0][0] = (*boundary_old)(bound,0);
        linear_bound[0][1] = (*boundary_old)(bound,1);

        // get current bound invertet
        linear_bound_inverted[0][0] = (*boundary_old)(bound,1);
        linear_bound_inverted[0][1] = (*boundary_old)(bound,0);

        // check wether current bound is in xy or yx coord in calculated midpoint vector
        auto index = std::find(already_calc_midpoints.begin(),
            already_calc_midpoints.end(), linear_bound[0]);
        if(index != already_calc_midpoints.end()) {
            // get midpoint index of current bound or invertetd bound
            new_bound = std::distance(already_calc_midpoints.begin(), index) + nodes_old->rows;
        } else {
            index = std::find(already_calc_midpoints.begin(),
                already_calc_midpoints.end(), linear_bound_inverted[0]);
            new_bound = std::distance(already_calc_midpoints.begin(), index) + nodes_old->rows;
        }

        // set midpoint in the middle of current bound
        quadratic_bound[0][2] = linear_bound[0][1];
        quadratic_bound[0][1] = new_bound;
        quadratic_bound[0][0] = linear_bound[0][0];

        // append current bound line with 3 indices to new boundary vector
        quadratic_boundary_vector.push_back(quadratic_bound[0]);
    }

    // create new element, node and boundary matrices
    auto nodes_new = std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::real>>(
        quadratic_node_vector.size(), 2, stream);
    auto elements_new = std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::index>>(
        quadratic_elements_vector.size(), 6, stream);
    auto boundary_new = std::make_shared<mpFlow::numeric::Matrix<mpFlow::dtype::index>>(
        quadratic_boundary_vector.size(), 3, stream);

    // copy element vector to element matrix
    for (mpFlow::dtype::index row = 0; row < quadratic_elements_vector.size(); ++row)
    for (mpFlow::dtype::index column = 0; column < quadratic_elements_vector[0].size(); ++column) {
        (*elements_new)(row, column) = quadratic_elements_vector[row][column];
    }

    // copy node vector to node matrix
    for (mpFlow::dtype::index row = 0; row < quadratic_node_vector.size(); ++row)
    for (mpFlow::dtype::index column = 0; column < quadratic_node_vector[0].size(); ++column) {
        (*nodes_new)(row, column) = quadratic_node_vector[row][column];
    }

    // copy boundary vector to boundary matrix
    for (mpFlow::dtype::index row = 0; row < quadratic_boundary_vector.size(); ++row)
    for (mpFlow::dtype::index column = 0; column < quadratic_boundary_vector[0].size(); ++column) {
        (*boundary_new)(row, column) = quadratic_boundary_vector[row][column];
    }
    nodes_new->copyToDevice(stream);
    elements_new->copyToDevice(stream);
    boundary_new->copyToDevice(stream);

    // return quadratic mesh matrices
    return std::make_tuple(nodes_new, elements_new, boundary_new);
}

std::tuple<
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic>,
    Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic>>
    mpFlow::numeric::irregularMesh::calculateGlobalEdgeIndices(
        std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements) {
    std::vector<std::tuple<mpFlow::dtype::index, mpFlow::dtype::index>> edges;
    Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic> globalEdgeIndex =
        Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>::Zero(elements->rows, 3);
    Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic> localEdges =
        Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>::Zero(elements->rows, 2 * 3);

    // find all unique edges
    for (dtype::index element = 0; element < elements->rows; ++element)
    for (dtype::index i = 0; i < 3; ++i) {
        // sort edge to guarantee constant global edge orientation
        auto edge = (*elements)(element, i) < (*elements)(element, (i + 1) % 3) ?
            std::make_tuple((*elements)(element, i), (*elements)(element, (i + 1) % 3)) :
            std::make_tuple((*elements)(element, (i + 1) % 3), (*elements)(element, i));

        // add edge to edges vector, if it was not already inserted
        auto edgePosition = std::find(edges.begin(), edges.end(), edge);
        if (edgePosition != edges.end()) {
            globalEdgeIndex(element, i) = std::distance(edges.begin(), edgePosition);
        }
        else {
            globalEdgeIndex(element, i) = edges.size();

            // add edge to vector
            edges.push_back(edge);
        }

        if ((*elements)(element, i) < (*elements)(element, (i + 1) % 3)) {
            localEdges(element, i * 2) = i;
            localEdges(element, i * 2 + 1) = (i + 1) % 3;
        }
        else {
            localEdges(element, i * 2) = (i + 1) % 3;
            localEdges(element, i * 2 + 1) = i;
        }
    }

    // convert edges vector to Eigen array
    Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic> edgesArray =
        Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>::Zero(edges.size(), 2);
    for (dtype::index i = 0; i < edges.size(); ++i) {
        edgesArray(i, 0) = std::get<0>(edges[i]);
        edgesArray(i, 1) = std::get<1>(edges[i]);
    }

    return std::make_tuple(edgesArray, globalEdgeIndex, localEdges);
}
