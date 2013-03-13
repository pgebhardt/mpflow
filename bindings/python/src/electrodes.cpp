#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

// create proper tuple object
tuple coordinates_wrapper(fastEIT::Electrodes* that, fastEIT::dtype::index index) {
    return make_tuple(
        make_tuple(
            std::get<0>(std::get<0>(that->coordinates(index))),
            std::get<1>(std::get<0>(that->coordinates(index)))),
        make_tuple(
            std::get<0>(std::get<1>(that->coordinates(index))),
            std::get<1>(std::get<1>(that->coordinates(index)))));
}

void pyfasteit::export_electrodes() {
    // declare some tuple converters
    create_tuple_converter<fastEIT::dtype::real, fastEIT::dtype::real>();

    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.electrodes"))));
    scope().attr("electrodes") = module;

    // expose Electrodes class
    class_<fastEIT::Electrodes,
        std::shared_ptr<fastEIT::Electrodes>>(
        "Electrodes",
        init<fastEIT::dtype::size,
            std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>,
            fastEIT::dtype::real>())
    .add_property("count", &fastEIT::Electrodes::count)
    .add_property("shape", &fastEIT::Electrodes::shape)
    .add_property("impedance", &fastEIT::Electrodes::impedance)
    .add_property("area", &fastEIT::Electrodes::area)
    .def("coordinates", &coordinates_wrapper);

    // set scope to sub module
    scope sub_module = module;

    def("circular_boundary", &fastEIT::electrodes::circularBoundary);

    // reset scope
    scope();
}
