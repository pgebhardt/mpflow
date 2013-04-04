#include <pyfasteit/pyfasteit.hpp>
#include <boost/python/slice.hpp>
#include <numpy/arrayobject.h>
using namespace boost::python;

template <
    class matrix_type,
    int npy_type
>
PyObject* numpy_array_from_matrix(fastEIT::Matrix<matrix_type>* matrix) {
    // create new numpy array
    npy_intp size[] = {
        (npy_intp)matrix->data_columns(),
        (npy_intp)matrix->data_rows()
    };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNewFromData(2, size, npy_type,
        matrix->host_data());

    // transpose and reshape array
    tuple shape = make_tuple(slice(_, matrix->rows()), slice(_, matrix->columns()));
    array = (PyArrayObject*)PyArray_Transpose(array, nullptr);
    array = (PyArrayObject*)PyObject_GetItem((PyObject*)array, shape.ptr());

    return (PyObject*)array;
}

template <
    class matrix_type,
    int npy_type
>
PyObject* get(fastEIT::Matrix<matrix_type>* self, cudaStream_t stream) {
    // get data from device and sync
    self->copyToHost(stream);
    cudaStreamSynchronize(stream);

    return numpy_array_from_matrix<matrix_type, npy_type>(self);
}

template <
    class matrix_type,
    int npy_type
>
PyObject* get_async(fastEIT::Matrix<matrix_type>* self, cudaStream_t stream) {
    // get data from device
    self->copyToHost(stream);

    return numpy_array_from_matrix<matrix_type, npy_type>(self);
}

template <
    class matrix_type,
    int npy_type
>
void put(fastEIT::Matrix<matrix_type>* self, numeric::array& numpy_array,
    cudaStream_t stream) {
    // check dimensions of numpy_array
    if (PyArray_NDIM((PyArrayObject*)numpy_array.ptr()) != 2) {
        throw std::invalid_argument("only array with ndim of 2 are allowed");
    }

    // convert array to correct data type
    npy_intp* shape = PyArray_DIMS((PyArrayObject*)numpy_array.ptr());
    if ((shape[0] != self->rows()) || (shape[1] != self->columns())) {
        std::stringstream message;
        message << "cannot assign array of incompatible shape (" << self->rows() << "," <<
            self->columns() << ") (" << shape[0] << "," << shape[1] << ")";
        throw std::invalid_argument(message.str());
    }

    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew(2, shape, npy_type);
    if (PyArray_CopyInto(array, (PyArrayObject*)numpy_array.ptr()) != 0) {
        throw std::invalid_argument("array of incompatible dtype");
    }

    // get strides and data
    npy_intp* strides = PyArray_STRIDES(array);
    char* data = (char*)PyArray_DATA(array);

    // fill matrix with data from numpy array
    for (fastEIT::dtype::index row = 0; row < self->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < self->columns(); ++column) {
        (*self)(row, column) = *reinterpret_cast<matrix_type*>(
            data + row * strides[0] + column * strides[1]);
    }
    self->copyToDevice(stream);
}

template <
    class matrix_type
>
tuple shape(fastEIT::Matrix<matrix_type>* self) {
    return make_tuple(self->rows(), self->columns());
}

template <
    class matrix_type,
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
    .add_property("shape", shape<matrix_type>)
    .def("copy", &fastEIT::Matrix<matrix_type>::copy)
    .def("copy_to_host", &fastEIT::Matrix<matrix_type>::copyToHost)
    .def("copy_to_device", &fastEIT::Matrix<matrix_type>::copyToDevice)
    .def("get", get<matrix_type, npy_type>)
    .def("get_async", get_async<matrix_type, npy_type>)
    .def("put", put<matrix_type, npy_type>);
}

template <
    class matrix_type,
    int npy_type
>
std::shared_ptr<fastEIT::Matrix<matrix_type>> fromNumpy(numeric::array& array,
    cudaStream_t stream) {
    // check dimension
    if (PyArray_NDIM(array.ptr()) != 2) {
        throw std::invalid_argument("only array with ndim of 2 are allowed");
    }

    // get array shape
    npy_intp* shape = PyArray_DIMS(array.ptr());

    // create fastEIT matrix with same shape as numpy array
    auto matrix = std::make_shared<fastEIT::Matrix<matrix_type>>(
        shape[0], shape[1], stream);

    // put array data to matrix
    put<matrix_type, npy_type>(matrix.get(), array, stream);

    return matrix;
}

void pyfasteit::export_matrix() {
    import_array();

    wrap_matrix<fastEIT::dtype::real, NPY_FLOAT32>("Matrix_real");
    wrap_matrix<fastEIT::dtype::index, NPY_UINT32>("Matrix_index");

    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.matrix"))));
    scope().attr("matrix") = module;
    scope sub_module = module;

    def("to_real", fromNumpy<fastEIT::dtype::real, NPY_FLOAT32>);
    def("to_index", fromNumpy<fastEIT::dtype::index, NPY_UINT32>);

    // reset scope
    scope();
}
