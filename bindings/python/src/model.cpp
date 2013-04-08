#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

void wrap_model(const char* name) {
    class_<fastEIT::Model_base,
        std::shared_ptr<fastEIT::Model_base>>(
        name, init<
            std::shared_ptr<fastEIT::Mesh>, std::shared_ptr<fastEIT::Electrodes>,
            std::shared_ptr<fastEIT::source::Source>, fastEIT::dtype::real,
            fastEIT::dtype::size>())
    .def("update", &fastEIT::Model_base::update)
    .add_property("mesh", &fastEIT::Model_base::mesh)
    .add_property("electrodes", &fastEIT::Model_base::electrodes)
    .add_property("source", &fastEIT::Model_base::source)
    .def("system_matrix", &fastEIT::Model_base::system_matrix)
    .def("potential", &fastEIT::Model_base::potential)
    .add_property("jacobian", &fastEIT::Model_base::jacobian)
    .add_property("s_matrix", &fastEIT::Model_base::s_matrix)
    .add_property("r_matrix", &fastEIT::Model_base::r_matrix)
    .add_property("connectivity_matrix", &fastEIT::Model_base::connectivity_matrix)
    .add_property("elemental_s_matrix", &fastEIT::Model_base::elemental_s_matrix)
    .add_property("elemental_r_matrix", &fastEIT::Model_base::elemental_r_matrix)
    .add_property("elemental_jacobian_matrix",
        &fastEIT::Model_base::elemental_jacobian_matrix)
    .add_property("sigma_ref", &fastEIT::Model_base::sigma_ref)
    .add_property("components_count", &fastEIT::Model_base::components_count);
}

template <
    class basis_function_type
>
void wrap_derived_model(const char* name) {
    class_<fastEIT::Model<basis_function_type>,
        std::shared_ptr<fastEIT::Model<basis_function_type>>,
        bases<fastEIT::Model_base>>(
        name, init<
            std::shared_ptr<fastEIT::Mesh>, std::shared_ptr<fastEIT::Electrodes>,
            std::shared_ptr<fastEIT::source::Source>, fastEIT::dtype::real,
            fastEIT::dtype::size, cublasHandle_t, cudaStream_t>());

    implicitly_convertible<fastEIT::Model<basis_function_type>,
        fastEIT::Model_base>();
    implicitly_convertible<std::shared_ptr<fastEIT::Model<basis_function_type>>,
        std::shared_ptr<fastEIT::Model_base>>();
}

void pyfasteit::export_model() {
    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.model"))));
    scope().attr("model") = module;
    scope sub_module = module;

    // wrap model base class
    wrap_model("Model");

    // wrap derived classes
    wrap_derived_model<fastEIT::basis::Linear>("Linear");
    wrap_derived_model<fastEIT::basis::Quadratic>("Quadratic");

    // reset scope
    scope();
}
