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
    Eigen::Ref<Eigen::ArrayXXi const> const elements, Eigen::Ref<Eigen::ArrayXXi const> const edges)
    : nodes(nodes), elements(elements), edges(distmesh::utils::fixBoundaryEdgeOrientation(nodes, elements, edges)),
    elementEdges(distmesh::utils::getTriangulationEdgeIndices(elements, this->edges)),
    boundary(distmesh::utils::boundEdges(elements, this->edges)) {
    // check input
    if (nodes.cols() != 2) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: nodes.cols() != 2");
    }
    if (elements.rows() == 0) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: elements.rows() == 0");
    }
    if (edges.rows() == 0) {
        throw std::invalid_argument("mpFlow::numeric::IrregularMesh::IrregularMesh: edges.rows() == 0");
    }
}

mpFlow::numeric::IrregularMesh::IrregularMesh(Eigen::Ref<Eigen::ArrayXXd const> const nodes,
    Eigen::Ref<Eigen::ArrayXXi const> const elements)
    : IrregularMesh(nodes, elements, distmesh::utils::findUniqueEdges(elements)) { }

std::shared_ptr<mpFlow::numeric::IrregularMesh> mpFlow::numeric::IrregularMesh::fromConfig(
    json_value const& config, json_value const& portConfig, cudaStream_t const stream, std::string const path) {
    // check for correct config
    if (config["height"].type == json_none) {
        return nullptr;
    }

    if (config["path"].type != json_none) {
        // load mesh from file
        std::string const meshPath = str::format("%s/%s")(path, std::string(config["path"]));

        auto const nodes = mpFlow::numeric::Matrix<double>::loadtxt(str::format("%s/nodes.txt")(meshPath), stream);
        auto const elements = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/elements.txt")(meshPath), stream);
        
        // try to load edges from file, but fallback to default edge implementation on error
        try {
            auto const edges = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/edges.txt")(meshPath), stream);
            return std::make_shared<mpFlow::numeric::IrregularMesh>(nodes->toEigen(), elements->toEigen(), edges->toEigen());            
        }
        catch(std::exception const&) {
            return std::make_shared<mpFlow::numeric::IrregularMesh>(nodes->toEigen(), elements->toEigen());            
        }
    }
    else if ((config["radius"].type != json_none) && (config["outerEdgeLength"].type != json_none) &&
            (config["innerEdgeLength"].type != json_none) && (portConfig.type != json_none)) {
        double const radius = config["radius"];
                
        // fix mesh at corner points of ports
        Eigen::ArrayXXd fixedPoints;
        if ((portConfig["width"].type != json_none) && (portConfig["count"].type != json_none)) {
            auto const width = portConfig["width"].u.dbl;
            auto const count = portConfig["count"].u.integer;
            auto const offset = portConfig["offset"].u.dbl;
            auto const invertDirection = portConfig["invertDirection"].u.boolean;
            
            fixedPoints = Eigen::ArrayXXd(2 * count, 2);
            fixedPoints << math::circularPoints(radius, 2.0 * M_PI * radius / count, offset, invertDirection),
                math::circularPoints(radius, 2.0 * M_PI * radius / count, offset + width, invertDirection);
        }
        
        // create mesh with libdistmesh
        auto const distanceFuntion = distmesh::distanceFunction::circular(radius);
        auto const distMesh = distmesh::distmesh(distanceFuntion, config["outerEdgeLength"],
            1.0 + (1.0 - (double)config["innerEdgeLength"] / (double)config["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::utils::boundingBox(2), fixedPoints);

        // create mpflow mesh objects from distmesh arrays
        auto const mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(std::get<0>(distMesh),
            std::get<1>(distMesh));

        // save mesh to files for later usage
        mkdir(str::format("%s/mesh")(path).c_str(), 0777);
        mpFlow::numeric::Matrix<double>::fromEigen(mesh->nodes, stream)->savetxt(str::format("%s/mesh/nodes.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->elements, stream)->savetxt(str::format("%s/mesh/elements.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->edges, stream)->savetxt(str::format("%s/mesh/edges.txt")(path));

        return mesh;
    }
    else {
        return nullptr;
    }
}

Eigen::ArrayXXd mpFlow::numeric::IrregularMesh::elementNodes(unsigned const element) const {
    // get node index and coordinate
    Eigen::ArrayXXd result(this->elements.cols(), 2);
    
    for (int node = 0; node < this->elements.cols(); ++node) {
        result.row(node) = this->nodes.row(this->elements(element, node));
    }

    return result;
}

Eigen::ArrayXXd mpFlow::numeric::IrregularMesh::edgeNodes(unsigned const edge) const {
    // get node index and coordinate
    Eigen::ArrayXXd result(this->edges.cols(), 2);
    
    for (int node = 0; node < this->edges.cols(); ++node) {
        result.row(node) = this->nodes.row(this->edges(edge, node));
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
    std::shared_ptr<IrregularMesh const> const mesh, cudaStream_t const stream) const {
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
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, double> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, thrust::complex<float>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Linear, thrust::complex<double>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, float> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, double> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, thrust::complex<float>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;
template std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>
    mpFlow::numeric::IrregularMesh::createInterpolationMatrix<mpFlow::FEM::basis::Quadratic, thrust::complex<double>> (
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const, cudaStream_t const) const;

/*
// create mesh for quadratic basis function
std::shared_ptr<mpFlow::numeric::IrregularMesh> mpFlow::numeric::irregularMesh::quadraticBasis(
    Eigen::Ref<Eigen::ArrayXXd const> const nodes,
    Eigen::Ref<Eigen::ArrayXXi const> const elements,
    Eigen::Ref<Eigen::ArrayXXi const> const boundary) {
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
*/