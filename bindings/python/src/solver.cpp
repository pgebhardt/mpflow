#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

void pyfasteit::export_solver() {
    // function pointer for overloaded methods
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver::*calibrate1)
        (cublasHandle_t, cudaStream_t) = &fastEIT::Solver::calibrate;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver::*calibrate2)
        (const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t) = &fastEIT::Solver::calibrate;

    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver::*solve1)
        (cublasHandle_t, cudaStream_t) = &fastEIT::Solver::solve;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver::*solve2)
        (const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t) = &fastEIT::Solver::solve;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver::*solve3)
        (const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t) = &fastEIT::Solver::solve;

    class_<fastEIT::Solver, std::shared_ptr<fastEIT::Solver>>(
        "Solver", init<std::shared_ptr<fastEIT::Model_base>,
            fastEIT::dtype::real, cublasHandle_t, cudaStream_t>())
    .def("pre_solve", &fastEIT::Solver::preSolve)
    .def("calibrate", calibrate1)
    .def("calibrate", calibrate2)
    .def("solve", solve1)
    .def("solve", solve2)
    .def("solve", solve3)
    .add_property("model", &fastEIT::Solver::model)
    .add_property("forward_solver", &fastEIT::Solver::forward_solver)
    .add_property("inverse_solver", &fastEIT::Solver::inverse_solver)
    .add_property("dgamma", &fastEIT::Solver::dgamma)
    .add_property("gamma", &fastEIT::Solver::gamma)
    .add_property("measured_voltage", &fastEIT::Solver::measured_voltage)
    .add_property("calibration_voltage", &fastEIT::Solver::calibration_voltage);
}

