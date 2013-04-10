#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

void wrap_source(const char* name) {
    class_<fastEIT::source::Source,
        std::shared_ptr<fastEIT::source::Source>>(
        name, init<std::string,
            fastEIT::dtype::real, std::shared_ptr<fastEIT::Mesh>,
            std::shared_ptr<fastEIT::Electrodes>, fastEIT::dtype::size,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t>())
    .def("update_excitation", &fastEIT::source::Source::updateExcitation)
    .add_property("drive_pattern", &fastEIT::source::Source::drive_pattern)
    .add_property("measurement_pattern", &fastEIT::source::Source::measurement_pattern)
    .add_property("pattern", &fastEIT::source::Source::pattern)
    .add_property("elemental_pattern", &fastEIT::source::Source::elemental_pattern)
    .add_property("d_matrix", &fastEIT::source::Source::d_matrix)
    .add_property("w_matrix", &fastEIT::source::Source::w_matrix)
    .add_property("x_matrix", &fastEIT::source::Source::x_matrix)
    .add_property("z_matrix", &fastEIT::source::Source::z_matrix)
    .def("excitation", &fastEIT::source::Source::excitation)
    .add_property("drive_count", &fastEIT::source::Source::drive_count)
    .add_property("measurement_count", &fastEIT::source::Source::measurement_count)
    .add_property("value", &fastEIT::source::Source::value);
}

template <
    template <class> class source_type,
    class basis_function_type
>
void wrap_derived_source(const char* name) {
    class_<source_type<basis_function_type>, std::shared_ptr<source_type<basis_function_type>>,
        bases<fastEIT::source::Source>>(
        name, init<
            fastEIT::dtype::real, std::shared_ptr<fastEIT::Mesh>,
            std::shared_ptr<fastEIT::Electrodes>, fastEIT::dtype::size,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t>());

    implicitly_convertible<source_type<basis_function_type>,
        fastEIT::source::Source>();
    implicitly_convertible<std::shared_ptr<source_type<basis_function_type>>,
        std::shared_ptr<fastEIT::source::Source>>();
}

void pyfasteit::export_source() {
    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.source"))));
    scope().attr("source") = module;
    scope sub_module = module;

    // wrap source base class
    wrap_source("Source");

    // create submodule for each type of source
    object current_module(handle<>(borrowed(PyImport_AddModule("fasteit.source.current"))));
    object voltage_module(handle<>(borrowed(PyImport_AddModule("fasteit.source.voltage"))));
    scope().attr("current") = current_module;
    scope().attr("voltage") = voltage_module;

    // wrap derived classes
    scope current_scope = current_module;
    wrap_derived_source<fastEIT::source::Current, fastEIT::basis::Linear>("Linear");
    wrap_derived_source<fastEIT::source::Current, fastEIT::basis::Quadratic>("Quadratic");

    scope voltage_scope = voltage_module;
    wrap_derived_source<fastEIT::source::Voltage, fastEIT::basis::Linear>("Linear");
    wrap_derived_source<fastEIT::source::Voltage, fastEIT::basis::Quadratic>("Quadratic");
}
