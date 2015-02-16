#include <chrono>
#include <iostream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
using namespace mpFlow;

#define RADIUS (0.085)

class Time {
private:
    std::chrono::high_resolution_clock::time_point time_;

public:
    Time() {
        this->restart();
    }
    void restart() {
        this->time_ = std::chrono::high_resolution_clock::now();
    }
    double elapsed() {
        return std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - this->time_).count();
    }
};

int main(int argc, char* argv[]) {
    Time time;

    // print out mpFlow version for refernce
    std::cout << "mpFlow version: " << version::getVersionString() << std::endl;

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    // cudaStreamCreate(&cudaStream);

    // create standard pattern
    auto drivePattern = std::make_shared<numeric::Matrix<dtype::real>>(36, 18, cudaStream);
    auto measurementPattern = std::make_shared<numeric::Matrix<dtype::real>>(36, 18, cudaStream);

    auto dist_mesh = distmesh::distmesh(distmesh::distance_function::circular(RADIUS),
        0.006, 1.0, RADIUS * 1.1 * distmesh::bounding_box(2));
    auto boundary = distmesh::boundedges(std::get<1>(dist_mesh));
    std::cout << "nodes: " << std::get<0>(dist_mesh).rows() << " elements: " << std::get<1>(dist_mesh).rows() << std::endl;

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
        FEM::sourceDescriptor::OpenSourceType, 1.0, electrodes,
        drivePattern, measurementPattern, cudaStream);

    // create equation
    auto equation = std::make_shared<FEM::Equation<dtype::real, FEM::basis::Linear>>(
        mesh, electrodes, 1.0, cudaStream);

    // benchmark different pipeline lengths
    std::array<dtype::index, 512> pipelineLengths;
    for (dtype::index i = 0; i < pipelineLengths.size(); ++i) {
        pipelineLengths[i] = i + 1;
    }

    auto data = std::make_shared<numeric::Matrix<dtype::real>>(pipelineLengths.size(), 2, cudaStream);
    for (size_t i = 0; i < pipelineLengths.size(); ++i) {
        // create solver
        auto solver = std::make_shared<EIT::Solver<FEM::basis::Linear>>(
            equation, source, 7, pipelineLengths[i], 0.0, cublasHandle, cudaStream);
        solver->preSolve(cublasHandle, cudaStream);

        cudaStreamSynchronize(cudaStream);
        time.restart();

        for (dtype::index i = 0; i < 10; ++i) {
            solver->solveDifferential(cublasHandle, cudaStream);
        }

        cudaStreamSynchronize(cudaStream);
        (*data)(i, 0) = (dtype::real)pipelineLengths[i];
        (*data)(i, 1) = (dtype::real)pipelineLengths[i] / (time.elapsed() / 10.0);

        std::cout << "parallelImages: " << pipelineLengths[i] << ", fps: " << (*data)(i, 1) << std::endl;
    }

    numeric::matrix::savetxt("fps_mbp.txt", data);

    return EXIT_SUCCESS;
}
