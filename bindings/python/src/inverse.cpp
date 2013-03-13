#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

fastEIT::dtype::real get_regularization_factor(
    fastEIT::InverseSolver<fastEIT::numeric::Conjugate>* that) {
    return that->regularization_factor();
}

void set_regularization_factor(fastEIT::InverseSolver<fastEIT::numeric::Conjugate>* that,
    fastEIT::dtype::real value) {
    that->regularization_factor() = value;
}

template void pyfasteit::export_inverse() {
    class_<fastEIT::InverseSolver<fastEIT::numeric::Conjugate>,
        std::shared_ptr<fastEIT::InverseSolver<fastEIT::numeric::Conjugate>>>(
        "InverseSolver", init<
            fastEIT::dtype::size, fastEIT::dtype::size,
            fastEIT::dtype::real, cublasHandle_t, cudaStream_t>())
    .def("solve", &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::solve)
    .def("calc_system_matrix",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::calcSystemMatrix)
    .def("calc_excitation",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::calcExcitation)
    .add_property("dvoltage",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::dvoltage)
    .add_property("zeros",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::zeros)
    .add_property("excitation",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::excitation)
    .add_property("system_matrix",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::system_matrix)
    .add_property("jacobian_square",
        &fastEIT::InverseSolver<fastEIT::numeric::Conjugate>::jacobian_square)
    .add_property("regularization_factor", &get_regularization_factor,
        &set_regularization_factor);
}

