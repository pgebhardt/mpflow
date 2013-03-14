#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

template <
    class model_type
>
void wrap_source(const char* name) {
    class_<fastEIT::source::Source<model_type>,
        std::shared_ptr<fastEIT::source::Source<model_type>>>(
        name, init<std::string,
            fastEIT::dtype::real, std::shared_ptr<model_type>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t>())
    .def("update_excitation", &fastEIT::source::Source<model_type>::updateExcitation)
    .add_property("model", &fastEIT::source::Source<model_type>::model)
    .add_property("drive_pattern", &fastEIT::source::Source<model_type>::drive_pattern)
    .add_property("measurement_pattern", &fastEIT::source::Source<model_type>::measurement_pattern)
    .add_property("pattern", &fastEIT::source::Source<model_type>::pattern)
    .add_property("elemental_pattern", &fastEIT::source::Source<model_type>::elemental_pattern)
    .add_property("excitation_matrix", &fastEIT::source::Source<model_type>::excitation_matrix)
    .def("excitation", &fastEIT::source::Source<model_type>::excitation)
    .add_property("drive_count", &fastEIT::source::Source<model_type>::drive_count)
    .add_property("measurement_count", &fastEIT::source::Source<model_type>::measurement_count)
    .add_property("value", &fastEIT::source::Source<model_type>::value);
}

template <
    template <class> class source_type,
    class model_type
>
void wrap_derived_source(const char* name) {
    class_<source_type<model_type>, std::shared_ptr<source_type<model_type>>,
        bases<fastEIT::source::Source<model_type>>>(
        name, init<
            fastEIT::dtype::real, std::shared_ptr<model_type>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            cublasHandle_t, cudaStream_t>());

    implicitly_convertible<source_type<model_type>, fastEIT::source::Source<model_type>>();
    implicitly_convertible<std::shared_ptr<source_type<model_type>>,
        std::shared_ptr<fastEIT::source::Source<model_type>>>();
}

void pyfasteit::export_source() {
    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.source"))));
    scope().attr("source") = module;
    scope sub_module = module;

    // wrap source base class
    wrap_source<fastEIT::Model<fastEIT::basis::Linear>>("Source");

    // wrap derived classes
    wrap_derived_source<fastEIT::source::Current,
        fastEIT::Model<fastEIT::basis::Linear>>("Current");
    wrap_derived_source<fastEIT::source::Voltage,
        fastEIT::Model<fastEIT::basis::Linear>>("Voltage");

    // reset scope
    scope();
}
