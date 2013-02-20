// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"
#include <iostream>

// create mesh class
template <
    class BasisFunction
>
fastEIT::Mesh<BasisFunction>::Mesh(std::shared_ptr<Matrix<dtype::real>> nodes,
    std::shared_ptr<Matrix<dtype::index>> elements, std::shared_ptr<Matrix<dtype::index>> boundary,
    dtype::real radius, dtype::real height)
    : nodes_(nodes), elements_(elements), boundary_(boundary), radius_(radius),
        height_(height) {
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

// create quadratic mesh
template <>
fastEIT::Mesh<fastEIT::basis::Quadratic>::Mesh(std::shared_ptr<Matrix<dtype::real>> nodes,
    std::shared_ptr<Matrix<dtype::index>> elements, std::shared_ptr<Matrix<dtype::index>> boundary,
    dtype::real radius, dtype::real height)
    : nodes_(nodes), elements_(elements), boundary_(boundary), radius_(radius),
        height_(height) {
    // check input
    if (radius <= 0.0f) {
        throw std::invalid_argument("radius <= 0.0");
    }
    if (height <= 0.0f) {
        throw std::invalid_argument("height <= 0.0");
    }
    if (elements->columns() != fastEIT::basis::Linear::nodes_per_element) {
        throw std::invalid_argument("elements.count() != BasisFunction::nodes_per_element");
    }
    if (boundary->columns() != fastEIT::basis::Linear::nodes_per_edge) {
        throw std::invalid_argument("boundary.count() != BasisFunction::nodes_per_edge");
    }

    std::tie(this->nodes_, this->elements_, this->boundary_) =
        mesh::quadraticMeshFromLinear(this->nodes(), this->elements(), this->boundary());
}

template <
    class BasisFunction
>
std::array<std::tuple<fastEIT::dtype::index, std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>>,
    BasisFunction::nodes_per_element> fastEIT::Mesh<BasisFunction>::elementNodes(dtype::index element) const {
    // result array
    std::array<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>, BasisFunction::nodes_per_element> result;

    // get node index and coordinate
    dtype::index index = -1;
    std::tuple<dtype::real, dtype::real> coordinates = std::make_tuple(0.0f, 0.0f);
    for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
        // get index
        index = (*this->elements())(element, node);

        // get coordinates
        coordinates = std::make_tuple((*this->nodes())(index, 0),
            (*this->nodes())(index, 1));

        // add to array
        result[node] = std::make_tuple(index, coordinates);
    }

    return result;
}

template <
    class BasisFunction
>
std::array<std::tuple<fastEIT::dtype::index, std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>>,
    BasisFunction::nodes_per_edge> fastEIT::Mesh<BasisFunction>::boundaryNodes(dtype::index element) const {
    // result array
    std::array<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>, BasisFunction::nodes_per_edge> result;

    // get node index and coordinate
    dtype::index index = -1;
    std::tuple<dtype::real, dtype::real> coordinates = std::make_tuple(0.0f, 0.0f);
    for (dtype::index node = 0; node < BasisFunction::nodes_per_edge; ++node) {
        // get index
        index = (*this->boundary())(element, node);

        // get coordinates
        coordinates = std::make_tuple((*this->nodes())(index, 0),
            (*this->nodes())(index, 1));

        // add to array
        result[node] = std::make_tuple(index, coordinates);
    }

    return result;
}

// function create quadratic mesh from linear
std::tuple<
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>>
fastEIT::mesh::quadraticMeshFromLinear(
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> nodes_old,
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> elements_old,
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> boundary_old) {
    // define vectors for calculation
    std::vector<std::array<fastEIT::dtype::index, 2> > already_calc_midpoints(0);
    std::vector<std::array<fastEIT::dtype::real, 2> > new_calc_nodes(0);
    std::vector<std::array<fastEIT::dtype::index, 6> > quadratic_elements_vector(elements_old->rows());
    std::vector<std::array<fastEIT::dtype::index, 2> > current_edge(2);
    std::vector<std::array<fastEIT::dtype::real, 2> > quadratic_node_vector(0);
    std::vector<std::array<fastEIT::dtype::real, 2> > node_from_linear(1);
    std::vector<std::array<fastEIT::dtype::index, 3> > quadratic_boundary_vector(0);
    std::vector<std::array<fastEIT::dtype::index, 3> > quadratic_bound(1);
    std::vector<std::array<fastEIT::dtype::index, 2> > linear_bound_inverted(1);
    std::vector<std::array<fastEIT::dtype::index, 2> > linear_bound(1);
    fastEIT::dtype::index new_bound = 0;

    // copy existing elements vom matrix to vector
    for (fastEIT::dtype::index element = 0; element < elements_old->rows(); ++element) {
        for (fastEIT::dtype::index element_node = 0; element_node < 3; ++element_node) {
            quadratic_elements_vector[element][element_node] = (*elements_old)(element, element_node);
        }
    }

    // calculate new midpoints between existing ones
    for (fastEIT::dtype::index element = 0; element < elements_old->rows(); ++element) {
        for (fastEIT::dtype::index element_node = 0; element_node < elements_old->columns(); ++element_node) {
            // get current edge
            current_edge[0][0] = (*elements_old)(element, element_node);
            current_edge[0][1] = (*elements_old)(element,
                (element_node + 1)%elements_old->columns());

            // get current edge inverted
            current_edge[1][0] = (*elements_old)(element,
                (element_node + 1)%elements_old->columns());
            current_edge[1][1] = (*elements_old)(element, element_node);

            //check if midpoint for current adge was calculated before
            auto index = std::find(already_calc_midpoints.begin(), already_calc_midpoints.end(), current_edge[0]);
            if(index != already_calc_midpoints.end()) {
                // midpoint already exists, using existing one. check for inverted coords too
                quadratic_elements_vector[element][element_node + 3] = nodes_old->rows() +
                    std::distance(already_calc_midpoints.begin(), index);
            } else {
                index = std::find(already_calc_midpoints.begin(),
                    already_calc_midpoints.end(), current_edge[1]);

                if(index != already_calc_midpoints.end()) {
                    quadratic_elements_vector[element][element_node + 3] = nodes_old->rows() +
                    std::distance(already_calc_midpoints.begin(), index);
                } else {
                    // midpoint does not exist. calculate new one
                    fastEIT::dtype::real node_x = 0.5 * ((*nodes_old)(current_edge[0][0], 0) +
                        (*nodes_old)(current_edge[0][1],0));
                    fastEIT::dtype::real node_y = 0.5 * ((*nodes_old)(current_edge[0][0], 1) +
                        (*nodes_old)(current_edge[0][1],1));

                    // create array with x and y coords
                    std::array<fastEIT::dtype::real, 2> midpoint = {{node_x, node_y}};

                    // append new midpoint to existing nodes
                    new_calc_nodes.push_back(midpoint);

                    // append current edge to 'already done' list
                    already_calc_midpoints.push_back(current_edge[0]);

                    // set new node for current element andd adjust node index
                    quadratic_elements_vector[element][element_node + 3] = 
                        (new_calc_nodes.size() - 1 + nodes_old->rows());
                }
            }
        }
    }

    // copy nodes from linear mesh and new nodes to one vector
    for (fastEIT::dtype::index node = 0; node < nodes_old->rows(); ++node) {
        node_from_linear[0][0] = (*nodes_old)(node, 0);
        node_from_linear[0][1] = (*nodes_old)(node, 1);
        quadratic_node_vector.push_back(node_from_linear[0]);
    }
    for (fastEIT::dtype::index new_node = 0; new_node < new_calc_nodes.size(); ++new_node) {
        quadratic_node_vector.push_back(new_calc_nodes[new_node]);
    }

    // calculate new boundary Matrix
    for(fastEIT::dtype::index bound = 0; bound < boundary_old->rows(); ++bound){
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
            new_bound = std::distance(already_calc_midpoints.begin(), index) + nodes_old->rows();
        } else {
            index = std::find(already_calc_midpoints.begin(),
                already_calc_midpoints.end(), linear_bound_inverted[0]);
            new_bound = std::distance(already_calc_midpoints.begin(), index) + nodes_old->rows();
        }

        // set midpoint in the middle of current bound
        quadratic_bound[0][2] = linear_bound[0][1];
        quadratic_bound[0][1] = new_bound;
        quadratic_bound[0][0] = linear_bound[0][0];

        // append current bound line with 3 indices to new boundary vector
        quadratic_boundary_vector.push_back(quadratic_bound[0]);
    }

    // create new element, node and boundary matrices
    auto nodes_new = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(
        quadratic_node_vector.size(), 2, nullptr);
    auto elements_new = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(
        quadratic_elements_vector.size(), 6, nullptr);
    auto boundary_new = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(
        quadratic_boundary_vector.size(), 3, nullptr);

    // copy element vector to element matrix
    for (fastEIT::dtype::index row = 0; row < quadratic_elements_vector.size(); ++row) {
        for (fastEIT::dtype::index column = 0; column < quadratic_elements_vector[0].size(); ++column) {
            (*elements_new)(row, column) = quadratic_elements_vector[row][column];
        }
    }
    // copy node vector to node matrix
    for (fastEIT::dtype::index row = 0; row < quadratic_node_vector.size(); ++row) {
        for (fastEIT::dtype::index column = 0; column < quadratic_node_vector[0].size(); ++column) {
            (*nodes_new)(row, column) = quadratic_node_vector[row][column];
        }
     }
    // copy boundary vector to boundary matrix
    for (fastEIT::dtype::index row = 0; row < quadratic_boundary_vector.size(); ++row) {
        for (fastEIT::dtype::index column = 0; column < quadratic_boundary_vector[0].size(); ++column) {
            (*boundary_new)(row, column) = quadratic_boundary_vector[row][column];
        }
     }

    // return quadratic mesh matrices
    return std::make_tuple(nodes_new, elements_new, boundary_new);
}

// class specialisations
template class fastEIT::Mesh<fastEIT::basis::Linear>;
template class fastEIT::Mesh<fastEIT::basis::Quadratic>;
