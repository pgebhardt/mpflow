#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

void wrap_model(const char* name) {
    class_<fastEIT::model::Model,
        std::shared_ptr<fastEIT::model::Model>>(
        name, init<
            std::shared_ptr<fastEIT::Mesh>, std::shared_ptr<fastEIT::Electrodes>,
            std::shared_ptr<fastEIT::source::Source>, fastEIT::dtype::real,
            fastEIT::dtype::size>())
    .def("update", &fastEIT::model::Model::update)
    .def("calc_jacobian", &fastEIT::model::Model::calcJacobian)
    .add_property("mesh", &fastEIT::model::Model::mesh)
    .add_property("electrodes", &fastEIT::model::Model::electrodes)
    .add_property("source", &fastEIT::model::Model::source)
    .def("system_matrix", &fastEIT::model::Model::system_matrix)
    .def("potential", &fastEIT::model::Model::potential)
    .add_property("jacobian", &fastEIT::model::Model::jacobian)
    .add_property("s_matrix", &fastEIT::model::Model::s_matrix)
    .add_property("r_matrix", &fastEIT::model::Model::r_matrix)
    .add_property("connectivity_matrix", &fastEIT::model::Model::connectivity_matrix)
    .add_property("elemental_s_matrix", &fastEIT::model::Model::elemental_s_matrix)
    .add_property("elemental_r_matrix", &fastEIT::model::Model::elemental_r_matrix)
    .add_property("elemental_jacobian_matrix",
        &fastEIT::model::Model::elemental_jacobian_matrix)
    .add_property("sigma_ref", &fastEIT::model::Model::sigma_ref)
    .add_property("component_count", &fastEIT::model::Model::component_count);
}

template <
    class basis_function_type
>
void wrap_derived_model(const char* name) {
    class_<fastEIT::Model<basis_function_type>,
        std::shared_ptr<fastEIT::Model<basis_function_type>>,
        bases<fastEIT::model::Model>>(
        name, init<
            std::shared_ptr<fastEIT::Mesh>, std::shared_ptr<fastEIT::Electrodes>,
            std::shared_ptr<fastEIT::source::Source>, fastEIT::dtype::real,
            fastEIT::dtype::size, cublasHandle_t, cudaStream_t>());

    implicitly_convertible<fastEIT::Model<basis_function_type>,
        fastEIT::model::Model>();
    implicitly_convertible<std::shared_ptr<fastEIT::Model<basis_function_type>>,
        std::shared_ptr<fastEIT::model::Model>>();
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
}
