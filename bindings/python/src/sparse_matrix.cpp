#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

template <
    class type
>
void wrap_sparse_matrix(const char* name) {
    class_<fastEIT::SparseMatrix<type>,
        std::shared_ptr<fastEIT::SparseMatrix<type>>>(
        name, init<
            fastEIT::dtype::size, fastEIT::dtype::size, cudaStream_t>())
    .def(init<std::shared_ptr<fastEIT::Matrix<type>>, cudaStream_t>())
    .def("to_matrix", &fastEIT::SparseMatrix<type>::toMatrix)
    .add_property("rows", &fastEIT::SparseMatrix<type>::rows)
    .add_property("columns", &fastEIT::SparseMatrix<type>::columns);
}

void pyfasteit::export_sparse_matrix() {
    wrap_sparse_matrix<fastEIT::dtype::real>("SparseMatrix_real");
    wrap_sparse_matrix<fastEIT::dtype::index>("SparseMatrix_index");
}
