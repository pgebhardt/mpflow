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
mpFlow::numeric::IrregularMesh::IrregularMesh(Eigen::Ref<Eigen::ArrayXXd const> const nodes,
    Eigen::Ref<Eigen::ArrayXXi const> const elements, Eigen::Ref<Eigen::ArrayXXi const> const boundary,
    double const height)
    : nodes(nodes), elements(elements), boundary(boundary), height(height) {
    // check input
    if (nodes.cols() != 2) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: nodes->cols != 2");
    }
    if (height <= 0.0) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: height <= 0.0");
    }
}

mpFlow::numeric::IrregularMesh::IrregularMesh(Eigen::Ref<Eigen::ArrayXXd const> const nodes,
    Eigen::Ref<Eigen::ArrayXXi const> const elements, double const height)
    : IrregularMesh(nodes, elements, distmesh::boundEdges(elements), height) { }

std::shared_ptr<mpFlow::numeric::IrregularMesh> mpFlow::numeric::IrregularMesh::fromConfig(
    json_value const& config,
    std::shared_ptr<FEM::BoundaryDescriptor const> const boundaryDescriptor,
    cudaStream_t const stream, std::string const path) {
    // check for correct config
    if (config["height"].type == json_none) {
        return nullptr;
    }

    // extract basic mesh parameter
    double const height = config["height"];

    if (config["path"].type != json_none) {
        // load mesh from file
        std::string const meshPath = str::format("%s/%s")(path, std::string(config["path"]));

        auto const nodes = mpFlow::numeric::Matrix<double>::loadtxt(str::format("%s/nodes.txt")(meshPath), stream);
        auto const elements = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/elements.txt")(meshPath), stream);
            
        return std::make_shared<mpFlow::numeric::IrregularMesh>(nodes->toEigen(), elements->toEigen(), height);
    }
    else if ((config["radius"].type != json_none) &&
            (config["outerEdgeLength"].type != json_none) &&
            (config["innerEdgeLength"].type != json_none)) {
        // fix mesh at boundaryDescriptor boundaries
        Eigen::ArrayXXd fixedPoints(boundaryDescriptor->count * 2, 2);
        for (unsigned i = 0; i < boundaryDescriptor->count; ++i) {
            fixedPoints.block(i * 2, 0, 1, 2) = boundaryDescriptor->coordinates.block(i, 0, 1, 2);
            fixedPoints.block(i * 2 + 1, 0, 1, 2) = boundaryDescriptor->coordinates.block(i, 2, 1, 2);
        }

        // create mesh with libdistmesh
        double const radius = config["radius"];
        auto const distanceFuntion = distmesh::distanceFunction::circular(radius);
        auto const distMesh = distmesh::distmesh(distanceFuntion, config["outerEdgeLength"],
            1.0 + (1.0 - (double)config["innerEdgeLength"] / (double)config["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::boundingBox(2), fixedPoints);

        // create mpflow matrix objects from distmesh arrays
        auto const mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(std::get<0>(distMesh),
            std::get<1>(distMesh), height);

        // save mesh to files for later usage
        mkdir(str::format("%s/mesh")(path).c_str(), 0777);
        mpFlow::numeric::Matrix<double>::fromEigen(mesh->nodes, stream)->savetxt(str::format("%s/mesh/nodes.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->elements, stream)->savetxt(str::format("%s/mesh/elements.txt")(path));

        return mesh;
    }
    else {
        return nullptr;
    }
}

Eigen::ArrayXXd mpFlow::numeric::IrregularMesh::elementNodes(unsigned const element) const {
    // result array
    Eigen::ArrayXXd result = Eigen::ArrayXXd::Zero(this->elements.cols(), 2);

    // get node index and coordinate
    for (int node = 0; node < this->elements.cols(); ++node) {
        result.row(node) = this->nodes.row(this->elements(element, node));
    }

    return result;
}

Eigen::ArrayXXd mpFlow::numeric::IrregularMesh::boundaryNodes(unsigned const element) const {
    // result array
    Eigen::ArrayXXd result = Eigen::ArrayXXd::Zero(this->boundary.cols(), 2);

    // get node index and coordinate
    for (int node = 0; node < this->boundary.cols(); ++node) {
        result.row(node) = this->nodes.row(this->boundary(element, node));
    }

    return result;
}

// interpolation
template <
    class interpolationFunctionType,
    class dataType
>
std::shared_ptr<mpFlow::numeric::SparseMatrix<dataType>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix(
    std::shared_ptr<IrregularMesh const> const mesh, cudaStream_t const stream) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh:createInterpolationMatrix: mesh == nullptr");
    }
    
    // evaluate interpolation function at node locations of the external mesh
    auto interpolationMatrix = std::make_shared<numeric::SparseMatrix<dataType>>(
        mesh->nodes.rows(), this->nodes.rows(), stream);
    for (int element = 0; element < this->elements.rows(); ++element) {
        // get points of element
        auto const points = this->elementNodes(element);
        
        // interpolate from each node point 
        for (int i = 0; i < points.rows(); ++i)
        for (int j = 0; j < mesh->nodes.rows(); ++j) {
            // evaluate interpolation function, only if point lies inside element
            if ((1.0 - 2.0 * distmesh::utils::pointsInsidePoly(mesh->nodes.row(j), points)(0)) < 0.0) {
                interpolationFunctionType const func(points, i);
                dataType const value = func.evaluate(mesh->nodes.row(j).transpose());
                
                interpolationMatrix->setValue(j, this->elements(element, i),
                    interpolationMatrix->getValue(j, this->elements(element, i)) + value);
            }
        }
    }
    interpolationMatrix->copyToDevice(stream);
    
    return interpolationMatrix;
}

template std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, float> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, double> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, thrust::complex<float>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, thrust::complex<double>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, float> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, double> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, thrust::complex<float>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, thrust::complex<double>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const);
    

// create mesh for quadratic basis function
std::shared_ptr<mpFlow::numeric::IrregularMesh> mpFlow::numeric::irregularMesh::quadraticBasis(
    Eigen::Ref<Eigen::ArrayXXd const> const nodes,
    Eigen::Ref<Eigen::ArrayXXi const> const elements,
    Eigen::Ref<Eigen::ArrayXXi const> const boundary,
    double const height) {
    // create quadratic grid
    Eigen::ArrayXXd n;
    Eigen::ArrayXXi e, b;
    std::tie(n, e, b) = quadraticMeshFromLinear(nodes, elements, boundary);

    // create mesh
    return std::make_shared<numeric::IrregularMesh>(n, e, height);
}

// function create quadratic mesh from linear
std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXi, Eigen::ArrayXXi>
mpFlow::numeric::irregularMesh::quadraticMeshFromLinear(
    Eigen::Ref<Eigen::ArrayXXd const> const nodes_old,
    Eigen::Ref<Eigen::ArrayXXi const> const elements_old,
    Eigen::Ref<Eigen::ArrayXXi const> const boundary_old) {
    // define vectors for calculation
    std::vector<std::array<unsigned, 2> > already_calc_midpoints(0);
    std::vector<std::array<double, 2> > new_calc_nodes(0);
    std::vector<std::array<unsigned, 6> > quadratic_elements_vector(
        elements_old.rows());
    std::vector<std::array<unsigned, 2> > current_edge(2);
    std::vector<std::array<double, 2> > quadratic_node_vector(0);
    std::vector<std::array<double, 2> > node_from_linear(1);
    std::vector<std::array<unsigned, 3> > quadratic_boundary_vector(0);
    std::vector<std::array<unsigned, 3> > quadratic_bound(1);
    std::vector<std::array<unsigned, 2> > linear_bound_inverted(1);
    std::vector<std::array<unsigned, 2> > linear_bound(1);
    unsigned new_bound = 0;

    // copy existing elements vom matrix to vector
    for (int element = 0; element < elements_old.rows(); ++element)
    for (int element_node = 0; element_node < 3; ++element_node) {
        quadratic_elements_vector[element][element_node] = elements_old(element, element_node);
    }

    // calculate new midpoints between existing ones
    for (int element = 0; element < elements_old.rows(); ++element)
    for (int element_node = 0; element_node < elements_old.cols();
        ++element_node) {
        // get current edge
        current_edge[0][0] = elements_old(element, element_node);
        current_edge[0][1] = elements_old(element,
            (element_node + 1)%elements_old.cols());

        // get current edge inverted
        current_edge[1][0] = elements_old(element,
            (element_node + 1)%elements_old.cols());
        current_edge[1][1] = elements_old(element, element_node);

        //check if midpoint for current adge was calculated before
        auto index = std::find(already_calc_midpoints.begin(), already_calc_midpoints.end(),
            current_edge[0]);
        if(index != already_calc_midpoints.end()) {
            // midpoint already exists, using existing one. check for inverted coords too
            quadratic_elements_vector[element][element_node + 3] = nodes_old.rows() +
                std::distance(already_calc_midpoints.begin(), index);
        } else {
            index = std::find(already_calc_midpoints.begin(),
                already_calc_midpoints.end(), current_edge[1]);

            if(index != already_calc_midpoints.end()) {
                quadratic_elements_vector[element][element_node + 3] = nodes_old.rows() +
                std::distance(already_calc_midpoints.begin(), index);
            } else {
                // midpoint does not exist. calculate new one
                double node_x = 0.5 * (nodes_old(current_edge[0][0], 0) +
                    nodes_old(current_edge[0][1],0));
                double node_y = 0.5 * (nodes_old(current_edge[0][0], 1) +
                    nodes_old(current_edge[0][1],1));

                // create array with x and y coords
                std::array<double, 2> midpoint = {{node_x, node_y}};

                // append new midpoint to existing nodes
                new_calc_nodes.push_back(midpoint);

                // append current edge to 'already done' list
                already_calc_midpoints.push_back(current_edge[0]);

                // set new node for current element andd adjust node index
                quadratic_elements_vector[element][element_node + 3] = 
                    (new_calc_nodes.size() - 1 + nodes_old.rows());
            }
        }
    }

    // copy nodes from linear mesh and new nodes to one vector
    for (int node = 0; node < nodes_old.rows(); ++node) {
        node_from_linear[0][0] = nodes_old(node, 0);
        node_from_linear[0][1] = nodes_old(node, 1);
        quadratic_node_vector.push_back(node_from_linear[0]);
    }
    for (unsigned new_node = 0; new_node < new_calc_nodes.size(); ++new_node) {
        quadratic_node_vector.push_back(new_calc_nodes[new_node]);
    }

    // calculate new boundary Matrix
    for (int bound = 0; bound < boundary_old.rows(); ++bound){
        // get current bound
        linear_bound[0][0] = boundary_old(bound,0);
        linear_bound[0][1] = boundary_old(bound,1);

        // get current bound invertet
        linear_bound_inverted[0][0] = boundary_old(bound,1);
        linear_bound_inverted[0][1] = boundary_old(bound,0);

        // check wether current bound is in xy or yx coord in calculated midpoint vector
        auto index = std::find(already_calc_midpoints.begin(),
            already_calc_midpoints.end(), linear_bound[0]);
        if(index != already_calc_midpoints.end()) {
            // get midpoint index of current bound or invertetd bound
            new_bound = std::distance(already_calc_midpoints.begin(), index) + nodes_old.rows();
        } else {
            index = std::find(already_calc_midpoints.begin(),
                already_calc_midpoints.end(), linear_bound_inverted[0]);
            new_bound = std::distance(already_calc_midpoints.begin(), index) + nodes_old.rows();
        }

        // set midpoint in the middle of current bound
        quadratic_bound[0][2] = linear_bound[0][1];
        quadratic_bound[0][1] = new_bound;
        quadratic_bound[0][0] = linear_bound[0][0];

        // append current bound line with 3 indices to new boundary vector
        quadratic_boundary_vector.push_back(quadratic_bound[0]);
    }

    // create new element, node and boundary matrices
    Eigen::ArrayXXd nodes_new = Eigen::ArrayXXd::Zero(quadratic_node_vector.size(), 2);
    Eigen::ArrayXXi elements_new = Eigen::ArrayXXi::Zero(quadratic_elements_vector.size(), 6);
    Eigen::ArrayXXi boundary_new = Eigen::ArrayXXi::Zero(quadratic_boundary_vector.size(), 3);

    // copy element vector to element matrix
    for (unsigned row = 0; row < quadratic_elements_vector.size(); ++row)
    for (unsigned column = 0; column < quadratic_elements_vector[0].size(); ++column) {
        elements_new(row, column) = quadratic_elements_vector[row][column];
    }

    // copy node vector to node matrix
    for (unsigned row = 0; row < quadratic_node_vector.size(); ++row)
    for (unsigned column = 0; column < quadratic_node_vector[0].size(); ++column) {
        nodes_new(row, column) = quadratic_node_vector[row][column];
    }

    // copy boundary vector to boundary matrix
    for (unsigned row = 0; row < quadratic_boundary_vector.size(); ++row)
    for (unsigned column = 0; column < quadratic_boundary_vector[0].size(); ++column) {
        boundary_new(row, column) = quadratic_boundary_vector[row][column];
    }

    // return quadratic mesh matrices
    return std::make_tuple(nodes_new, elements_new, boundary_new);
}

std::tuple<Eigen::ArrayXXi,
    std::vector<std::array<std::tuple<unsigned, std::tuple<unsigned, unsigned>>, 3>>>
    mpFlow::numeric::irregularMesh::calculateGlobalEdgeIndices(
        Eigen::Ref<Eigen::ArrayXXi const> const elements) {
    std::vector<std::tuple<unsigned, unsigned>> edges;
    std::vector<std::array<std::tuple<unsigned, std::tuple<unsigned, unsigned>>, 3>> localEdgeConnections(elements.rows());

    // find all unique edges
    for (int element = 0; element < elements.rows(); ++element)
    for (unsigned i = 0; i < 3; ++i) {
        // sort edge to guarantee constant global edge orientation
        auto const edge = elements(element, i) < elements(element, (i + 1) % 3) ?
            std::make_tuple(elements(element, i), elements(element, (i + 1) % 3)) :
            std::make_tuple(elements(element, (i + 1) % 3), elements(element, i));

        // add edge to edges vector, if it was not already inserted
        auto const edgePosition = std::find(edges.begin(), edges.end(), edge);
        if (edgePosition != edges.end()) {
            localEdgeConnections[element][i] = std::make_tuple(std::distance(edges.begin(), edgePosition),
                elements(element, i) < elements(element, (i + 1) % 3) ?
                std::make_tuple(i, (i + 1) % 3) : std::make_tuple((i + 1) % 3, i));
        }
        else {
            edges.push_back(edge);
            localEdgeConnections[element][i] = std::make_tuple(edges.size() - 1,
                elements(element, i) < elements(element, (i + 1) % 3) ?
                std::make_tuple(i, (i + 1) % 3) : std::make_tuple((i + 1) % 3, i));
        }
    }

    // convert stl vectors to eigen arrays
    Eigen::ArrayXXi edgesArray(edges.size(), 2);
    for (unsigned edge = 0; edge < edges.size(); ++edge) {
        edgesArray(edge, 0) = std::get<0>(edges[edge]);
        edgesArray(edge, 1) = std::get<1>(edges[edge]);
    }

    return std::make_tuple(edgesArray, localEdgeConnections);
}
