#include <pyfasteit/pyfasteit.hpp>
#include <numpy/arrayobject.h>
using namespace boost::python;

template <
    class type,
    int npy_type
>
numeric::array get(fastEIT::Matrix<type>* that, cudaStream_t stream) {
    that->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // create new numpy array
    npy_intp size[] = {
        (int)that->rows(),
        (int)that->columns()
    };
    PyObject* obj = PyArray_SimpleNew(2, size, npy_type);
    handle<> h(obj);
    numeric::array array(h);

    // fill array with data
    for (fastEIT::dtype::index row = 0; row < that->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < that->columns(); ++column) {
        array[make_tuple(row, column)] = (*that)(row, column);
    }

    return array;
}

template <
    class type,
    int npy_type
>
void wrap_matrix(const char* name) {
    class_<fastEIT::Matrix<type>,
        std::shared_ptr<fastEIT::Matrix<type>>>(
        name, init<fastEIT::dtype::size, fastEIT::dtype::size,
            cudaStream_t>())
    .def(init<fastEIT::dtype::size, fastEIT::dtype::size,
            cudaStream_t, type>())
    .add_property("rows", &fastEIT::Matrix<type>::rows)
    .add_property("columns", &fastEIT::Matrix<type>::columns)
    .def("copy", &fastEIT::Matrix<type>::copy)
    .def("copy_to_host", &fastEIT::Matrix<type>::copyToHost)
    .def("copy_to_device", &fastEIT::Matrix<type>::copyToDevice)
    .def("get", get<type, npy_type>);
}

template <
    class matrix_type,
    class cast_type,
    int npy_type
>
std::shared_ptr<fastEIT::Matrix<matrix_type>> fromNumpy(numeric::array& numpy_array,
    cudaStream_t stream) {
    // cast array to npy_type
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OF(
        numpy_array.ptr(), npy_type);

    // check success and dims
    if ((array == nullptr) || (PyArray_NDIM(array) != 2)) {
        return nullptr;
    }

    // get array shape, strides and data
    npy_intp* shape = PyArray_DIMS(array);
    npy_intp* strides = PyArray_STRIDES(array);
    char* data = (char*)PyArray_DATA(array);

    // create fastEIT matrix with same shape as numpy array
    auto matrix = std::make_shared<fastEIT::Matrix<matrix_type>>(
        shape[0], shape[1], stream);

    // fill matrix with data from numpy array
    for (fastEIT::dtype::index row = 0; row < matrix->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < matrix->columns(); ++column) {
        (*matrix)(row, column) = *reinterpret_cast<cast_type*>(
            data + row * strides[0] + column * strides[1]);
    }
    matrix->copyToDevice(stream);

    return matrix;
}

void pyfasteit::export_matrix() {
    import_array();

    wrap_matrix<fastEIT::dtype::real, NPY_FLOAT32>("Matrix_real");
    wrap_matrix<fastEIT::dtype::index, NPY_UINT>("Matrix_index");

    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.matrix"))));
    scope().attr("matrix") = module;
    scope sub_module = module;

    def("to_real", fromNumpy<fastEIT::dtype::real, double, NPY_DOUBLE>);
    def("to_index", fromNumpy<fastEIT::dtype::index, long, NPY_LONG>);

    // reset scope
    scope();
}
