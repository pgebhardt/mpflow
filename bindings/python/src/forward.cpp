#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

template <
    class model_type
>
void wrap_forward(const char* name) {
    class_<fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate, model_type>,
        std::shared_ptr<fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
            model_type>>>(
        name, init<
            std::shared_ptr<model_type>, cublasHandle_t, cudaStream_t>())
    .def("apply_measurement_pattern",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::applyMeasurementPattern)
    .def("solve",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::solve)
    .add_property("model",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::model)
    .add_property("voltage",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::voltage)
    .add_property("current",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::current);
}

void pyfasteit::export_forward() {
    wrap_forward<fastEIT::Model<fastEIT::basis::Linear>>("ForwardSolver");
}

