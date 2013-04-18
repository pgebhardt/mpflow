#include <pyfasteit/pyfasteit.hpp>
#include <numpy/arrayobject.h>
using namespace boost::python;

std::shared_ptr<fastEIT::Mesh> CreateMeshFromNumpyArrays(numeric::array& nodes,
    numeric::array& elements, numeric::array& boundary, fastEIT::dtype::real radius,
    fastEIT::dtype::real height, cudaStream_t stream) {
    // create gpu matrices from arrays
    auto gpu_nodes = pyfasteit::fromNumpy<fastEIT::dtype::real, NPY_FLOAT32>(
        nodes, stream);
    auto gpu_elements = pyfasteit::fromNumpy<fastEIT::dtype::index, NPY_UINT32>(
        elements, stream);
    auto gpu_boundary = pyfasteit::fromNumpy<fastEIT::dtype::index, NPY_UINT32>(
        boundary, stream);

    return std::make_shared<fastEIT::Mesh>(gpu_nodes, gpu_elements, gpu_boundary,
        radius, height);
}

std::shared_ptr<fastEIT::Mesh> quadraticBasis_wrapper(numeric::array& nodes,
    numeric::array& elements, numeric::array& boundary, fastEIT::dtype::real radius,
    fastEIT::dtype::real height, cudaStream_t stream) {
    // create gpu matrices from arrays
    auto gpu_nodes = pyfasteit::fromNumpy<fastEIT::dtype::real, NPY_FLOAT32>(
        nodes, stream);
    auto gpu_elements = pyfasteit::fromNumpy<fastEIT::dtype::index, NPY_UINT32>(
        elements, stream);
    auto gpu_boundary = pyfasteit::fromNumpy<fastEIT::dtype::index, NPY_UINT32>(
        boundary, stream);

    return fastEIT::mesh::quadraticBasis(gpu_nodes, gpu_elements, gpu_boundary,
        radius, height, stream);
}

void pyfasteit::export_mesh() {
    import_array();

    class_<fastEIT::Mesh,
        std::shared_ptr<fastEIT::Mesh>>(
        "Mesh", init<
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
            fastEIT::dtype::real, fastEIT::dtype::real>())
    .def("__init__", make_constructor(&CreateMeshFromNumpyArrays))
    .add_property("nodes", &fastEIT::Mesh::nodes)
    .add_property("elements", &fastEIT::Mesh::elements)
    .add_property("boundary", &fastEIT::Mesh::boundary)
    .add_property("radius", &fastEIT::Mesh::radius)
    .add_property("height", &fastEIT::Mesh::height);

    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.mesh"))));
    scope().attr("mesh") = module;
    scope sub_module = module;

    def("quadratic_basis", &fastEIT::mesh::quadraticBasis);
    def("quadratic_basis", &quadraticBasis_wrapper);
}
