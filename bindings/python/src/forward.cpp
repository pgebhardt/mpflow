#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

void pyfasteit::export_forward() {
    class_<fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>,
        std::shared_ptr<fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>>>(
        "ForwardSolver", init<
            std::shared_ptr<fastEIT::Model_base>, cublasHandle_t, cudaStream_t>())
    .def("apply_measurement_pattern",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>::applyMeasurementPattern)
    .def("solve",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>::solve)
    .add_property("model",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>::model)
    .add_property("voltage",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>::voltage)
    .add_property("current",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>::current);
}

