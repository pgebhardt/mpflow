#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"

using namespace mpFlow;
#define RADIUS (0.085)

int main(int argc, char* argv[]) {
    HighPrecisionTime time;

    // print out mpFlow version for refernce
    str::print(str::format("mpFlow version: %s")(version::getVersionString()));

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // create model classes corresponding to ERT 2.0 measurement system
    // create standard pattern
    auto drivePattern = std::make_shared<numeric::Matrix<dtype::real>>(36, 18, cudaStream);
    auto measurementPattern = std::make_shared<numeric::Matrix<dtype::real>>(36, 18, cudaStream);
    for (dtype::index i = 0; i < drivePattern->cols; ++i) {
        // simple ERT 2.0 pattern
        (*drivePattern)(i * 2, i) = 1.0 - (i % 2) * 2.0;
        (*drivePattern)(((i + 1) * 2) % drivePattern->rows, i) = -1.0 + (i % 2) * 2.0;
    }
    for (dtype::index i = 0; i < measurementPattern->cols; ++i) {
        // simple ERT 2.0 pattern
        (*measurementPattern)(i * 2 + 1, i) = 1.0 - (i % 2) * 2.0;
        (*measurementPattern)(((i + 1) * 2 + 1) % measurementPattern->rows, i) = -1.0 + (i % 2) * 2.0;
    }
    drivePattern->copyToDevice(cudaStream);
    measurementPattern->copyToDevice(cudaStream);

    // Create Mesh using libdistmesh
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create mesh using libdistmesh with uniform grid size");

    auto dist_mesh = distmesh::distmesh(distmesh::distance_function::circular(RADIUS),
        0.006, 1.0, RADIUS * 1.1 * distmesh::bounding_box(2));
    auto boundary = distmesh::boundedges(std::get<1>(dist_mesh));

    str::print(str::format("Mesh created with %d nodes and %d elements")(
        std::get<0>(dist_mesh).rows(), std::get<1>(dist_mesh).rows()));
    str::print(str::format("Time: %f ms")(time.elapsed() * 1e3));

    // create mpflow mesh object
    auto mesh = std::make_shared<numeric::IrregularMesh>(
        numeric::matrix::fromEigen<dtype::real, distmesh::dtype::real>(std::get<0>(dist_mesh), cudaStream),
        numeric::matrix::fromEigen<dtype::index, distmesh::dtype::index>(std::get<1>(dist_mesh), cudaStream),
        numeric::matrix::fromEigen<dtype::index, distmesh::dtype::index>(boundary, cudaStream), RADIUS, 0.4);

    // create electrodes
    auto electrodes = FEM::boundaryDescriptor::circularBoundary(
        36, std::make_tuple(0.005, 0.005), RADIUS, 0.0);

    // create source
    auto source = std::make_shared<FEM::SourceDescriptor>(
        FEM::SourceDescriptor::Type::Open, 1.0, electrodes,
        drivePattern, measurementPattern, cudaStream);

    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create equation model class");

    // create equation
    auto equation = std::make_shared<FEM::Equation<dtype::real, FEM::basis::Linear>>(
        mesh, electrodes, 1.0, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print(str::format("Time: %f ms")(time.elapsed() * 1e3));

    // benchmark different pipeline lengths
    std::array<dtype::index, 512> pipelineLengths;
    for (dtype::index i = 0; i < pipelineLengths.size(); ++i) {
        pipelineLengths[i] = i + 1;
    }

    // Create solver class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main solver class");

    auto solver = std::make_shared<EIT::Solver<FEM::basis::Linear, numeric::ConjugateGradient>>(
        equation, source, 7, 1, 0.0, cublasHandle, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print(str::format("Time: %f ms")(time.elapsed() * 1e3));

    // initialize solver
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Solve forward model and initialize inverse solver matrices");

    solver->preSolve(cublasHandle, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print(str::format("Time: %f ms")(time.elapsed() * 1e3));

    str::print("----------------------------------------------------");
    str::print("Reconstruct images for different pipeline lengths:\n");

    for (size_t i = 0; i < pipelineLengths.size(); ++i) {
        // create inverse solver
        solver = std::make_shared<EIT::Solver<FEM::basis::Linear, numeric::ConjugateGradient>>(
            equation, source, 7, pipelineLengths[i], 0.0, cublasHandle, cudaStream);

        cudaStreamSynchronize(cudaStream);
        time.restart();

        for (dtype::index i = 0; i < 10; ++i) {
            solver->solveDifferential(cublasHandle, cudaStream);
        }

        cudaStreamSynchronize(cudaStream);
        str::print(str::format("pipeline length: %d, time: %f ms, fps: %f")(
            pipelineLengths[i], time.elapsed() / 10.0 * 1e3,
            (dtype::real)pipelineLengths[i] / (time.elapsed() / 10.0)));
    }

    return EXIT_SUCCESS;
}
