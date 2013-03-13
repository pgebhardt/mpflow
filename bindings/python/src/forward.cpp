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
            std::shared_ptr<model_type>,
            std::shared_ptr<fastEIT::source::Source<model_type>>,
            cublasHandle_t, cudaStream_t>())
    .def("apply_measurement_pattern",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::applyMeasurementPattern)
    .def("solve",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::solve)
    .add_property("model",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::model)
    .add_property("source",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::source)
    .add_property("jacobian",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::jacobian)
    .add_property("voltage",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::voltage)
    .add_property("current",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::current)
    .def("potential",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::potential)
    .add_property("elemental_jacobian_matrix",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::elemental_jacobian_matrix)
    .add_property("electrode_attachment_matrix",
        &fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
        model_type>::electrode_attachment_matrix);
}

void pyfasteit::export_forward() {
    wrap_forward<fastEIT::Model<fastEIT::basis::Linear>>("ForwardSolver");
}

