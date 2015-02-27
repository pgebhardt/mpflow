#include <iostream>
#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "utils/stringtools/format.hpp"
#include "utils/high_precision_time.h"
#include "utils/json.c"

using namespace mpFlow;

// helper function to create an mpflow matrix from an json array
template <class type>
std::shared_ptr<numeric::Matrix<type>> matrixFromJsonArray(const json_value& array, cudaStream_t cudaStream) {
    // exctract sizes
    dtype::size rows = array.u.array.length;
    dtype::size cols = array[0].type == json_array ? array[0].u.array.length : 1;

    // create matrix
    auto matrix = std::make_shared<numeric::Matrix<type>>(rows, cols, cudaStream);

    // exctract values
    if (array[0].type != json_array) {
        for (dtype::index row = 0; row < matrix->rows; ++row) {
            (*matrix)(row, 0) = array[row].u.dbl;
        }
    }
    else {
        for (dtype::index row = 0; row < matrix->rows; ++row)
        for (dtype::index col = 0; col < matrix->cols; ++col) {
            (*matrix)(row, col) = array[row][col].u.dbl;
        }
    }
    matrix->copyToDevice(cudaStream);

    return matrix;
}

// helper function to create unit matrix
template <class type>
std::shared_ptr<numeric::Matrix<type>> eye(dtype::index size, cudaStream_t cudaStream) {
    auto matrix = std::make_shared<numeric::Matrix<type>>(size, size, cudaStream);
    for (dtype::index i = 0; i < size; ++i) {
        (*matrix)(i, i) = 1;
    }
    matrix->copyToDevice(cudaStream);

    return matrix;
}

void setCircularRegion(float x, float y, float radius,
    float value, std::shared_ptr<numeric::IrregularMesh> mesh,
    std::shared_ptr<numeric::Matrix<dtype::real>> gamma) {
    for (dtype::index element = 0; element < mesh->elements->rows; ++element) {
        auto nodes = mesh->elementNodes(element);

        dtype::real midX = 0.0, midY = 0.0;
        for (dtype::index node = 0; node < nodes.size(); ++node) {
            midX += std::get<0>(std::get<1>(nodes[node])) / nodes.size();
            midY += std::get<1>(std::get<1>(nodes[node])) / nodes.size();
        }
        if (math::square(midX - x) + math::square(midY - y) <= math::square(radius)) {
            (*gamma)(element, 0) = value;
        }
    }
}

int main(int argc, char* argv[]) {
    HighPrecisionTime time;

    // print out mpFlow version for refernce
    std::cout << "mpFlow version: " << version::getVersionString() << std::endl;

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // load config document
    if (argc <= 1) {
        std::cout << "You need to give modelLoader a path to the model config" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Load model from config file: " << argv[1] << std::endl;

    std::ifstream file(argv[1]);
    std::string fileContent((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    auto config = json_parse(fileContent.c_str(), fileContent.length());

    // check for success
    if (config == nullptr) {
        std::cout << "Error: Cannot parse config file" << std::endl;
        return EXIT_FAILURE;
    }

    // extract model config
    auto modelConfig = (*config)["model"];
    if (modelConfig.type == json_none) {
        std::cout << "Error: Invalid model config" << std::endl;
        return EXIT_FAILURE;
    }

    // Create Mesh using libdistmesh
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;

    // get mesh config
    auto meshConfig = modelConfig["mesh"];
    if (meshConfig.type == json_none) {
        std::cout << "Error: Invalid model config" << std::endl;
        return EXIT_FAILURE;
    }

    // create mesh from config or load from files, if mesh dir is given
    std::shared_ptr<numeric::IrregularMesh> mesh = nullptr;
    double radius = meshConfig["radius"];

    // create electrodes descriptor
    auto electrodes = FEM::boundaryDescriptor::circularBoundary(modelConfig["electrodes"]["count"].u.integer,
        std::make_tuple(modelConfig["electrodes"]["width"].u.dbl, modelConfig["electrodes"]["height"].u.dbl),
        radius, 0.0);

    if (meshConfig["mesh_dir"].type != json_none) {
        // load mesh from file
        std::string meshDir(meshConfig["meshPath"]);
        std::cout << "Load mesh from files: " << meshDir << std::endl;

        auto nodes = numeric::matrix::loadtxt<dtype::real>(str::format("%s/nodes.txt")(meshDir), cudaStream);
        auto elements = numeric::matrix::loadtxt<dtype::index>(str::format("%s/elements.txt")(meshDir), cudaStream);
        auto boundary = numeric::matrix::loadtxt<dtype::index>(str::format("%s/boundary.txt")(meshDir), cudaStream);
        mesh = std::make_shared<numeric::IrregularMesh>(nodes, elements, boundary, radius, (double)meshConfig["height"]);

        std::cout << "Mesh loaded with " << nodes->rows << " nodes and " <<
            elements->rows << " elements." << std::endl;
        std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;
    }
    else {
        std::cout << "Create mesh using libdistmesh" << std::endl;

        // fix mesh at electrodes boundaries
        distmesh::dtype::array<distmesh::dtype::real> fixedPoints(electrodes->count * 2, 2);
        for (dtype::index electrode = 0; electrode < electrodes->count; ++electrode) {
            fixedPoints(electrode * 2, 0) = std::get<0>(std::get<0>(electrodes->coordinates[electrode]));
            fixedPoints(electrode * 2, 1) = std::get<1>(std::get<0>(electrodes->coordinates[electrode]));
            fixedPoints(electrode * 2 + 1, 0) = std::get<0>(std::get<1>(electrodes->coordinates[electrode]));
            fixedPoints(electrode * 2 + 1, 1) = std::get<1>(std::get<1>(electrodes->coordinates[electrode]));
        }

        // create mesh with libdistmesh
        auto distanceFuntion = distmesh::distance_function::circular(radius);
        auto dist_mesh = distmesh::distmesh(distanceFuntion, meshConfig["outerEdgeLength"],
            1.0 + (1.0 - (double)meshConfig["innerEdgeLength"] / (double)meshConfig["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::bounding_box(2), fixedPoints);

        std::cout << "Mesh created with " << std::get<0>(dist_mesh).rows() << " nodes and " <<
            std::get<1>(dist_mesh).rows() << " elements." << std::endl;
        std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

        // create mpflow matrix objects from distmesh arrays
        auto nodes = numeric::matrix::fromEigen<dtype::real, distmesh::dtype::real>(std::get<0>(dist_mesh));
        auto elements = numeric::matrix::fromEigen<dtype::index, distmesh::dtype::index>(std::get<1>(dist_mesh));
        auto boundary = numeric::matrix::fromEigen<dtype::index, distmesh::dtype::index>(distmesh::boundedges(std::get<1>(dist_mesh)));
        mesh = std::make_shared<numeric::IrregularMesh>(nodes, elements, boundary, radius, (double)meshConfig["height"]);
    }

    // Create model helper classes
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Create model helper classes" << std::endl;

    // load excitation and measurement pattern from config or assume standard pattern, if not given
    std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern = nullptr;
    if (modelConfig["source"]["drivePattern"].type != json_none) {
        drivePattern = matrixFromJsonArray<dtype::real>(modelConfig["source"]["drivePattern"], cudaStream);
    }
    else {
        drivePattern = eye<dtype::real>(electrodes->count, cudaStream);;
    }

    std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern = nullptr;
    if (modelConfig["source"]["measurementPattern"].type != json_none) {
        measurementPattern = matrixFromJsonArray<dtype::real>(modelConfig["source"]["measurementPattern"], cudaStream);
    }
    else {
        measurementPattern = eye<dtype::real>(electrodes->count, cudaStream);;
    }

    // read out currents
    std::vector<dtype::real> excitation(drivePattern->cols);
    if (modelConfig["source"]["value"].type == json_array) {
        for (dtype::index i = 0; i < drivePattern->cols; ++i) {
            excitation[i] = modelConfig["source"]["value"][i].u.dbl;
        }
    }
    else {
        excitation = std::vector<dtype::real>(drivePattern->cols, (dtype::real)modelConfig["source"]["value"].u.dbl);
    }

    // create source descriptor
    auto sourceType = std::string(modelConfig["source"]["type"]) == "voltage" ?
        FEM::SourceDescriptor::Type::Fixed : FEM::SourceDescriptor::Type::Open;
    auto source = std::make_shared<FEM::SourceDescriptor>(sourceType, excitation, electrodes,
        drivePattern, measurementPattern, cudaStream);

    cudaStreamSynchronize(cudaStream);
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // Create main model class
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Create main model class" << std::endl;

    auto equation = std::make_shared<FEM::Equation<dtype::real, FEM::basis::Linear>>(
        mesh, electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    cudaStreamSynchronize(cudaStream);
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // Create forward solver and solve potential
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Solve electrical potential for all excitations"  << std::endl;

    auto gamma = std::make_shared<numeric::Matrix<dtype::real>>(mesh->elements->rows, 1, cudaStream);

    setCircularRegion(0.0, 0.015, 0.005, 3.0, mesh, gamma);
    gamma->copyToDevice(cudaStream);

    std::shared_ptr<numeric::Matrix<dtype::real>> result = nullptr;

    // use different numeric solver for different source types
    if (sourceType == FEM::SourceDescriptor::Type::Fixed) {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<FEM::basis::Linear, numeric::BiCGSTAB>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        time.restart();
        result = forwardSolver->solve(gamma, mesh->nodes->rows, cublasHandle, cudaStream, 1e-12);
    }
    else {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<FEM::basis::Linear, numeric::ConjugateGradient>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        time.restart();
        result = forwardSolver->solve(gamma, mesh->nodes->rows, cublasHandle, cudaStream, 1e-12);
    }

    cudaStreamSynchronize(cudaStream);
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // Print result
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Result:" << std::endl;
    numeric::matrix::savetxt(result, &std::cout);
    numeric::matrix::savetxt("result.txt", result);

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
