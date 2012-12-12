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

// create quadratic mesh
template <>
fastEIT::Mesh<fastEIT::basis::Quadratic>::Mesh(std::shared_ptr<Matrix<dtype::real>> nodes,
    std::shared_ptr<Matrix<dtype::index>> elements, std::shared_ptr<Matrix<dtype::index>> boundary,
    dtype::real radius, dtype::real height)
    : radius_(radius), height_(height), nodes_(nodes), elements_(elements),
        boundary_(boundary) {
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

std::tuple<
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>>
fastEIT::mesh::quadraticMeshFromLinear(
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> nodes_old,
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> elements_old,
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>> boundary_old) {
    std::vector<std::array<fastEIT::dtype::index, 2> > indices(0);
    std::vector<std::array<fastEIT::dtype::real, 2> > pnew(0);

    std::vector<std::array<fastEIT::dtype::index, 6> > tnew(elements_old->rows());
    std::vector<std::array<fastEIT::dtype::index, 2> > edge(2);

    for (fastEIT::dtype::index element = 0; element < elements_old->rows(); ++element) {
        // copy
        for (fastEIT::dtype::index element_node = 0; element_node < 3; ++element_node) {
            tnew[element][element_node] = (*elements_old)(element, element_node);
        }
    }
    //calc edge
    for (fastEIT::dtype::index element =0; element < elements_old->rows(); ++element) {
     for (fastEIT::dtype::index element_node =0; element_node < elements_old->columns(); ++element_node) {
        edge[0][0] = (*elements_old)(element, element_node);
        edge[0][1] = (*elements_old)(element, (element_node+1)%elements_old->columns());
        edge[1][0] = (*elements_old)(element, (element_node+1)%elements_old->columns());
        edge[1][1] = (*elements_old)(element, element_node);
    //check if edge in list
        auto index = std::find(indices.begin(), indices.end(), edge[0]);
        if(index != indices.end()) {
       // edge in indices
            tnew[element][element_node+3]=nodes_old->rows() + std::distance(indices.begin(), index);
        } else {
            index = std::find(indices.begin(), indices.end(), edge[1]);
            if(index != indices.end()) {
                tnew[element][element_node+3]=nodes_old->rows() + std::distance(indices.begin(), index);
            } else {
            // edge not in indices 
                fastEIT::dtype::real node_x = 0.5 * ((*nodes_old)(edge[0][0], 0) + (*nodes_old)(edge[0][1],0));
                fastEIT::dtype::real node_y = 0.5 * ((*nodes_old)(edge[0][0], 1) + (*nodes_old)(edge[0][1],1));
                std::array<fastEIT::dtype::real, 2> midpoint = {node_x , node_y};
                pnew.push_back(midpoint);
                indices.push_back(edge[0]);
                tnew[element][element_node+3]=(pnew.size()-1+nodes_old->rows());
                }
            }
        }
    }
    // vstack nodes_old and pnew

    std::vector<std::array<fastEIT::dtype::real, 2> > p(0);
    std::vector<std::array<fastEIT::dtype::real, 2> > old_node(1);

    for (fastEIT::dtype::index node =0; node < nodes_old->rows(); ++node) {
        old_node[0][0] = (*nodes_old)(node, 0);
        old_node[0][1] = (*nodes_old)(node, 1);
        p.push_back(old_node[0]);
     }


    for (fastEIT::dtype::index new_node=0; new_node < pnew.size(); ++new_node) {
        p.push_back(pnew[new_node]);
    }
    // adjust boundary Matrix

    std::vector<std::array<fastEIT::dtype::index, 3> > bnew(0);
    fastEIT::dtype::index new_bound=0;
    std::vector<std::array<fastEIT::dtype::index, 3> >new_bound_line(1);
    std::vector<std::array<fastEIT::dtype::index, 2> >bound_old_inv(1);
    std::vector<std::array<fastEIT::dtype::index, 2> >bound_old(1);

    for(fastEIT::dtype::index bound = 0; bound < boundary_old->rows(); ++bound){
        bound_old[0][0]=(*boundary_old)(bound,0);
        bound_old[0][1]=(*boundary_old)(bound,1);
        bound_old_inv[0][0]=(*boundary_old)(bound,1);
        bound_old_inv[0][1]=(*boundary_old)(bound,0);
        auto index = std::find(indices.begin(), indices.end(), bound_old[0]);
            if(index != indices.end()) {
                new_bound=std::distance(indices.begin(), index)+nodes_old->rows();
        } else {
                index = std::find(indices.begin(), indices.end(), bound_old_inv[0]);
                new_bound=std::distance(indices.begin(), index)+nodes_old->rows();
        }
        new_bound_line[0][2]=bound_old[0][1];
        new_bound_line[0][1]=new_bound;
        new_bound_line[0][0]=bound_old[0][0];
        bnew.push_back(new_bound_line[0]);
    }

    // create matrices
    auto nodes_new = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(p.size(), 2, nullptr);
    auto elements_new = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(tnew.size(), 6, nullptr);
    auto boundary_new = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(bnew.size(), 3, nullptr);

    // copy vectors to matrices

    for (fastEIT::dtype::index row=0; row < tnew.size(); ++row) {
        for (fastEIT::dtype::index column=0; column < tnew[0].size(); ++column) {
            (*elements_new)(row, column) = tnew[row][column];
        }
    }

    for (fastEIT::dtype::index row =0; row < p.size(); ++row) {
        for (fastEIT::dtype::index column=0; column < p[0].size(); ++column) {
            (*nodes_new)(row, column) = p[row][column];
        }
     }

    for (fastEIT::dtype::index row =0; row < bnew.size(); ++row) {
        for (fastEIT::dtype::index column=0; column < bnew[0].size(); ++column) {
            (*boundary_new)(row, column) = bnew[row][column];
        }
     }

    return std::make_tuple(nodes_new, elements_new, boundary_new);
}

// class specialisations
template class fastEIT::Mesh<fastEIT::basis::Linear>;
template class fastEIT::Mesh<fastEIT::basis::Quadratic>;
