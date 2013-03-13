#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

void pyfasteit::export_mesh() {
    class_<fastEIT::Mesh,
        std::shared_ptr<fastEIT::Mesh>>(
        "Mesh", init<
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
            std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>,
            fastEIT::dtype::real, fastEIT::dtype::real>())
    .add_property("nodes", &fastEIT::Mesh::nodes)
    .add_property("elements", &fastEIT::Mesh::elements)
    .add_property("boundary", &fastEIT::Mesh::boundary)
    .add_property("radius", &fastEIT::Mesh::radius)
    .add_property("height", &fastEIT::Mesh::height);
}
