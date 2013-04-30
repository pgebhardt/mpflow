#include <pyfasteit/pyfasteit.hpp>
#include <numpy/arrayobject.h>
#include <functional>
using namespace boost::python;

template <
    class source_type
>
std::shared_ptr<source_type> CreateSourceFromNumpy(fastEIT::dtype::real value,
    std::shared_ptr<fastEIT::Mesh> mesh, std::shared_ptr<fastEIT::Electrodes> electrodes,
    fastEIT::dtype::size component_count, numeric::array& drive_pattern,
    numeric::array& measurement_pattern, cublasHandle_t handle, cudaStream_t stream) {
    // create gpu matrices from arrays
    auto gpu_drive_pattern = pyfasteit::fromNumpy<fastEIT::dtype::real, NPY_FLOAT32>(
        drive_pattern, stream);
    auto gpu_measurement_pattern = pyfasteit::fromNumpy<fastEIT::dtype::real, NPY_FLOAT32>(
        measurement_pattern, stream);

    return std::make_shared<source_type>(value, mesh, electrodes, component_count,
        gpu_drive_pattern, gpu_measurement_pattern, handle, stream);
}

PyObject* values_getter(fastEIT::source::Source* self) {
    // create new numpy array
    npy_intp size[] = {
        (npy_intp)self->values().size(),
    };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(1, size, NPY_FLOAT32,
        self->values().data());

    return (PyObject*)array;
}

void values_setter(fastEIT::source::Source* self, numeric::array& value) {
    for (fastEIT::dtype::index excitation = 0; excitation < self->drive_count(); ++excitation) {
        self->values()[excitation] = extract<fastEIT::dtype::real>(value[excitation]);
    }
}

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
    .add_property("d_matrix", &fastEIT::source::Source::d_matrix)
    .add_property("w_matrix", &fastEIT::source::Source::w_matrix)
    .add_property("x_matrix", &fastEIT::source::Source::x_matrix)
    .add_property("z_matrix", &fastEIT::source::Source::z_matrix)
    .def("excitation", &fastEIT::source::Source::excitation)
    .add_property("drive_count", &fastEIT::source::Source::drive_count)
    .add_property("measurement_count", &fastEIT::source::Source::measurement_count)
    .add_property("values", &values_getter, &values_setter)
    .add_property("component_count", &fastEIT::source::Source::component_count);
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
            cublasHandle_t, cudaStream_t>())
    .def("__init__", make_constructor(&CreateSourceFromNumpy<source_type<basis_function_type>>));

    implicitly_convertible<source_type<basis_function_type>,
        fastEIT::source::Source>();
    implicitly_convertible<std::shared_ptr<source_type<basis_function_type>>,
        std::shared_ptr<fastEIT::source::Source>>();
}

void pyfasteit::export_source() {
    import_array();

    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.source"))));
    scope().attr("source") = module;
    scope sub_module = module;

    // wrap source base class
    wrap_source("Source");

    // expose linear sources as standard sources

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
