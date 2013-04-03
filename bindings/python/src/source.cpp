#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

template <
    class basis_function_type
>
void wrap_source(const char* name) {
    class_<fastEIT::source::Source<basis_function_type>,
        std::shared_ptr<fastEIT::source::Source<basis_function_type>>>(
        name, init<std::string,
            fastEIT::dtype::real, std::shared_ptr<fastEIT::Mesh>,
            std::shared_ptr<fastEIT::Electrodes>, fastEIT::dtype::size,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t>())
    .def("update_excitation", &fastEIT::source::Source<basis_function_type>::updateExcitation)
    .add_property("drive_pattern", &fastEIT::source::Source<basis_function_type>::drive_pattern)
    .add_property("measurement_pattern",
        &fastEIT::source::Source<basis_function_type>::measurement_pattern)
    .add_property("pattern", &fastEIT::source::Source<basis_function_type>::pattern)
    .add_property("elemental_pattern",
        &fastEIT::source::Source<basis_function_type>::elemental_pattern)
    .add_property("d_matrix", &fastEIT::source::Source<basis_function_type>::d_matrix)
    .add_property("w_matrix", &fastEIT::source::Source<basis_function_type>::w_matrix)
    .add_property("x_matrix", &fastEIT::source::Source<basis_function_type>::x_matrix)
    .add_property("z_matrix", &fastEIT::source::Source<basis_function_type>::z_matrix)
    .def("excitation", &fastEIT::source::Source<basis_function_type>::excitation)
    .add_property("drive_count", &fastEIT::source::Source<basis_function_type>::drive_count)
    .add_property("measurement_count",
        &fastEIT::source::Source<basis_function_type>::measurement_count)
    .add_property("value", &fastEIT::source::Source<basis_function_type>::value);
}

template <
    template <class> class source_type,
    class basis_function_type
>
void wrap_derived_source(const char* name) {
    class_<source_type<basis_function_type>, std::shared_ptr<source_type<basis_function_type>>,
        bases<fastEIT::source::Source<basis_function_type>>>(
        name, init<
            fastEIT::dtype::real, std::shared_ptr<fastEIT::Mesh>,
            std::shared_ptr<fastEIT::Electrodes>, fastEIT::dtype::size,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t>());

    implicitly_convertible<source_type<basis_function_type>,
        fastEIT::source::Source<basis_function_type>>();
    implicitly_convertible<std::shared_ptr<source_type<basis_function_type>>,
        std::shared_ptr<fastEIT::source::Source<basis_function_type>>>();
}

void pyfasteit::export_source() {
    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.source"))));
    scope().attr("source") = module;
    scope sub_module = module;

    // wrap source base class
    wrap_source<fastEIT::basis::Linear>("Source");

    // wrap derived classes
    wrap_derived_source<fastEIT::source::Current,
        fastEIT::basis::Linear>("Current");
    wrap_derived_source<fastEIT::source::Voltage,
        fastEIT::basis::Linear>("Voltage");

    // reset scope
    scope();
}
