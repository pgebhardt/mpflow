#include <iostream>
#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "utils/high_precision_time.h"
#include "utils/json.c"

using namespace mpFlow;

// helper function to create an mpflow matrix from an json array
// json arrays are assumed to be 2 dimensional
template <class type>
std::shared_ptr<numeric::Matrix<type>> matrixFromJsonArray(const json_value& array, cudaStream_t cudaStream) {
    // create matrix
    auto matrix = std::make_shared<numeric::Matrix<type>>(array.u.array.length, array[0].u.array.length, cudaStream);

    // exctract values
    for (dtype::index row = 0; row < matrix->rows; ++row)
    for (dtype::index col = 0; col < matrix->cols; ++col) {
        (*matrix)(row, col) = array[row][col].u.dbl;
    }
    matrix->copyToDevice(cudaStream);

    return matrix;
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
    if (modelConfig.type == json_null) {
        std::cout << "Error: Invalid model config" << std::endl;
        return EXIT_FAILURE;
    }

    // Create Mesh using libdistmesh
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Create mesh using libdistmesh" << std::endl;

    // get mesh config
    auto meshConfig = modelConfig["mesh"];
    if (meshConfig.type == json_null) {
        std::cout << "Error: Invalid model config" << std::endl;
        return EXIT_FAILURE;
    }

    // create mesh
    double radius = meshConfig["radius"];
    auto distanceFuntion = distmesh::distance_function::circular(radius);
    auto dist_mesh = distmesh::distmesh(distanceFuntion, meshConfig["outer_edge_length"],
        1.0 + (1.0 - (double)meshConfig["inner_edge_length"] / (double)meshConfig["outer_edge_length"]) *
        distanceFuntion / radius, 1.1 * radius * distmesh::bounding_box(2));

    std::cout << "Mesh created with " << std::get<0>(dist_mesh).rows() << " nodes and " <<
        std::get<1>(dist_mesh).rows() << " elements." << std::endl;
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // create mpflow matrix objects from distmesh arrays
    auto nodes = numeric::matrix::fromEigen<dtype::real, distmesh::dtype::real>(std::get<0>(dist_mesh));
    auto elements = numeric::matrix::fromEigen<dtype::index, distmesh::dtype::index>(std::get<1>(dist_mesh));
    auto boundary = numeric::matrix::fromEigen<dtype::index, distmesh::dtype::index>(distmesh::boundedges(std::get<1>(dist_mesh)));
    auto mesh = std::make_shared<numeric::IrregularMesh>(nodes, elements, boundary, radius, (double)meshConfig["height"]);

    // Create model helper classes
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Create model helper classes" << std::endl;

    // create electrodes descriptor
    auto electrodes = FEM::boundaryDescriptor::circularBoundary(modelConfig["electrodes"]["count"].u.integer,
        std::make_tuple(modelConfig["electrodes"]["width"].u.dbl, modelConfig["electrodes"]["height"].u.dbl),
        mesh->radius, 0.0);

    // load excitation and measurement pattern from config
    auto drivePattern = matrixFromJsonArray<dtype::real>(modelConfig["source"]["drive_pattern"], cudaStream);
    auto measurementPattern = matrixFromJsonArray<dtype::real>(modelConfig["source"]["measurement_pattern"], cudaStream);

    // read out currents
    std::vector<dtype::real> current(drivePattern->cols);
    if (modelConfig["source"]["current"].type == json_array) {
        for (dtype::index i = 0; i < drivePattern->cols; ++i) {
            current[i] = modelConfig["source"]["current"][i].u.dbl;
        }
    }
    else {
        current = std::vector<dtype::real>(drivePattern->cols, (dtype::real)modelConfig["source"]["current"].u.dbl);
    }

    // create source descriptor
    auto source = std::make_shared<FEM::SourceDescriptor>(FEM::sourceDescriptor::OpenSourceType,
        current, electrodes, drivePattern, measurementPattern, cudaStream);

    cudaStreamSynchronize(cudaStream);
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // Create main model class
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Create main model class" << std::endl;

    auto equation = std::make_shared<FEM::Equation<dtype::real, FEM::basis::Linear>>(
        mesh, electrodes, modelConfig["sigma_ref"].u.dbl, cudaStream);

    cudaStreamSynchronize(cudaStream);
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // Create forward solver and solve potential
    time.restart();
    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Solve electrical potential for all excitations"  << std::endl;

    auto forwardSolver = std::make_shared<EIT::ForwardSolver<FEM::basis::Linear, numeric::ConjugateGradient>>(
        equation, source, modelConfig["components_count"].u.integer, cublasHandle, cudaStream);
    auto gamma = std::make_shared<numeric::Matrix<dtype::real>>(mesh->elements->rows, 1, cudaStream);

    time.restart();
    auto result = forwardSolver->solve(gamma, mesh->nodes->rows, cublasHandle, cudaStream);

    cudaStreamSynchronize(cudaStream);
    std::cout << "Time: " << time.elapsed() * 1e3 << " ms" << std::endl;

    // Print result
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);

    std::cout << "----------------------------------------------------" << std::endl;
    std::cout << "Result:" << std::endl;
    numeric::matrix::savetxt(result, &std::cout);

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
