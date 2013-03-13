#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

BOOST_PYTHON_MODULE(fasteit) {
    // declare this module as a package
    scope().attr("__path__") = "fasteit";

    // use numpy ndarray as numeric type
    numeric::array::set_module_and_type("numpy", "ndarray");

    // export modules
    pyfasteit::export_cuda();
    pyfasteit::export_matrix();
    pyfasteit::export_sparse_matrix();
    pyfasteit::export_electrodes();
    pyfasteit::export_mesh();
    pyfasteit::export_model();
    pyfasteit::export_source();
    pyfasteit::export_forward();
    pyfasteit::export_inverse();
    pyfasteit::export_solver();
}
