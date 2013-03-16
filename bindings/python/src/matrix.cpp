#include <pyfasteit/pyfasteit.hpp>
#include <numpy/arrayobject.h>
using namespace boost::python;

template <
    class matrix_type,
    int npy_type
>
numeric::array get(fastEIT::Matrix<matrix_type>* self, cudaStream_t stream) {
    self->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // create new numpy array
    npy_intp size[] = {
        (int)self->rows(),
        (int)self->columns()
    };
    PyObject* obj = PyArray_SimpleNew(2, size, npy_type);
    handle<> h(obj);
    numeric::array array(h);

    // fill array with data
    for (fastEIT::dtype::index row = 0; row < self->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < self->columns(); ++column) {
        array[make_tuple(row, column)] = (*self)(row, column);
    }

    return array;
}

template <
    class matrix_type,
    class cast_type,
    int npy_type
>
void put(fastEIT::Matrix<matrix_type>* self, numeric::array& numpy_array,
    cudaStream_t stream) {
    // cast array to npy_type
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OF(
        numpy_array.ptr(), npy_type);

    // check success and dims
    if ((array == nullptr) || (PyArray_NDIM(array) != 2)) {
        // TODO: raise an exception
        return;
    }

    // get array shape, strides and data
    npy_intp* shape = PyArray_DIMS(array);
    npy_intp* strides = PyArray_STRIDES(array);
    char* data = (char*)PyArray_DATA(array);

    // check shape
    if ((shape[0] != self->rows()) || (shape[1] != self->columns())) {
        // TODO: raise an exception
        return;
    }

    // fill matrix with data from numpy array
    for (fastEIT::dtype::index row = 0; row < self->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < self->columns(); ++column) {
        (*self)(row, column) = *reinterpret_cast<cast_type*>(
            data + row * strides[0] + column * strides[1]);
    }
    self->copyToDevice(stream);
}

template <
    class matrix_type,
    class cast_type,
    int npy_type
>
void wrap_matrix(const char* name) {
    class_<fastEIT::Matrix<matrix_type>,
        std::shared_ptr<fastEIT::Matrix<matrix_type>>>(
        name, init<fastEIT::dtype::size, fastEIT::dtype::size,
            cudaStream_t>())
    .def(init<fastEIT::dtype::size, fastEIT::dtype::size,
            cudaStream_t, matrix_type>())
    .add_property("rows", &fastEIT::Matrix<matrix_type>::rows)
    .add_property("columns", &fastEIT::Matrix<matrix_type>::columns)
    .def("copy", &fastEIT::Matrix<matrix_type>::copy)
    .def("copy_to_host", &fastEIT::Matrix<matrix_type>::copyToHost)
    .def("copy_to_device", &fastEIT::Matrix<matrix_type>::copyToDevice)
    .def("get", get<matrix_type, npy_type>)
    .def("put", put<matrix_type, cast_type, npy_type>);
}

template <
    class matrix_type,
    class cast_type,
    int npy_type
>
std::shared_ptr<fastEIT::Matrix<matrix_type>> fromNumpy(numeric::array& array,
    cudaStream_t stream) {
    // check dimension
    if (PyArray_NDIM(array.ptr()) != 2) {
        return nullptr;
    }

    // get array shape
    npy_intp* shape = PyArray_DIMS(array.ptr());

    // create fastEIT matrix with same shape as numpy array
    auto matrix = std::make_shared<fastEIT::Matrix<matrix_type>>(
        shape[0], shape[1], stream);

    // put array data to matrix
    put<matrix_type, cast_type, npy_type>(matrix.get(), array, stream);

    return matrix;
}

void pyfasteit::export_matrix() {
    import_array();

    wrap_matrix<fastEIT::dtype::real, double, NPY_DOUBLE>("Matrix_real");
    wrap_matrix<fastEIT::dtype::index, long, NPY_LONG>("Matrix_index");

    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.matrix"))));
    scope().attr("matrix") = module;
    scope sub_module = module;

    def("to_real", fromNumpy<fastEIT::dtype::real, double, NPY_DOUBLE>);
    def("to_index", fromNumpy<fastEIT::dtype::index, long, NPY_LONG>);

    // reset scope
    scope();
}
