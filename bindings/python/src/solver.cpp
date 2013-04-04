#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

template <
    class model_type
>
void wrap_solver(const char* name) {
    // function pointer for overloaded methods
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver<model_type>::*calibrate1)
        (cublasHandle_t, cudaStream_t) = &fastEIT::Solver<model_type>::calibrate;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver<model_type>::*calibrate2)
        (const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t) = &fastEIT::Solver<model_type>::solve;

    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver<model_type>::*solve1)
        (cublasHandle_t, cudaStream_t) = &fastEIT::Solver<model_type>::solve;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver<model_type>::*solve2)
        (const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t) = &fastEIT::Solver<model_type>::solve;
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
        (fastEIT::Solver<model_type>::*solve3)
        (const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t) = &fastEIT::Solver<model_type>::solve;

    class_<fastEIT::Solver<model_type>,
        std::shared_ptr<fastEIT::Solver<model_type>>>(
        name, init<std::shared_ptr<model_type>,
            fastEIT::dtype::real, cublasHandle_t, cudaStream_t>())
    .def("pre_solve", &fastEIT::Solver<model_type>::preSolve)
    .def("calibrate", calibrate1)
    .def("calibrate", calibrate2)
    .def("solve", solve1)
    .def("solve", solve2)
    .def("solve", solve3)
    .add_property("model", &fastEIT::Solver<model_type>::model)
    .add_property("forward_solver", &fastEIT::Solver<model_type>::forward_solver)
    .add_property("inverse_solver", &fastEIT::Solver<model_type>::inverse_solver)
    .add_property("dgamma", &fastEIT::Solver<model_type>::dgamma)
    .add_property("gamma", &fastEIT::Solver<model_type>::gamma)
    .add_property("measured_voltage", &fastEIT::Solver<model_type>::measured_voltage)
    .add_property("calibration_voltage", &fastEIT::Solver<model_type>::calibration_voltage);
}

void pyfasteit::export_solver() {
    wrap_solver<fastEIT::Model<fastEIT::basis::Linear>>("Solver");
}

