#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

template <
    class basis_function_type
>
void wrap_model(const char* name) {
    class_<fastEIT::Model<basis_function_type>,
        std::shared_ptr<fastEIT::Model<basis_function_type>>>(
        name, init<
            std::shared_ptr<fastEIT::Mesh>, std::shared_ptr<fastEIT::Electrodes>,
            std::shared_ptr<fastEIT::source::Source<basis_function_type>>,
            fastEIT::dtype::real, fastEIT::dtype::size, cublasHandle_t, cudaStream_t>())
    .def("update", &fastEIT::Model<basis_function_type>::update)
    .add_property("mesh", &fastEIT::Model<basis_function_type>::mesh)
    .add_property("electrodes", &fastEIT::Model<basis_function_type>::electrodes)
    .add_property("source", &fastEIT::Model<basis_function_type>::source)
    .def("system_matrix", &fastEIT::Model<basis_function_type>::system_matrix)
    .def("potential", &fastEIT::Model<basis_function_type>::potential)
    .add_property("jacobian", &fastEIT::Model<basis_function_type>::jacobian)
    .add_property("s_matrix", &fastEIT::Model<basis_function_type>::s_matrix)
    .add_property("r_matrix", &fastEIT::Model<basis_function_type>::r_matrix)
    .add_property("connectivity_matrix", &fastEIT::Model<basis_function_type>::connectivity_matrix)
    .add_property("elemental_s_matrix", &fastEIT::Model<basis_function_type>::elemental_s_matrix)
    .add_property("elemental_r_matrix", &fastEIT::Model<basis_function_type>::elemental_r_matrix)
    .add_property("elemental_jacobian_matrix",
        &fastEIT::Model<basis_function_type>::elemental_jacobian_matrix)
    .add_property("sigma_ref", &fastEIT::Model<basis_function_type>::sigma_ref)
    .add_property("components_count", &fastEIT::Model<basis_function_type>::components_count);
}

void pyfasteit::export_model() {
    wrap_model<fastEIT::basis::Linear>("Model");
}
