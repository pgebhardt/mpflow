#include <pyfasteit/pyfasteit.hpp>
using namespace boost::python;

cudaStream_t cudaStreamCreate_wrapper() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}

cublasHandle_t cublasCreate_wrapper() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    return handle;
}

void cudaStreamSynchronize_wrapper(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

void pyfasteit::export_cuda() {
    // expose this module as part of fasteit package
    object module(handle<>(borrowed(PyImport_AddModule("fasteit.cuda"))));
    scope().attr("cuda") = module;
    scope sub_module = module;

    def("stream", &cudaStreamCreate_wrapper,
        return_value_policy<return_opaque_pointer>());
    def("cublas_handle", &cublasCreate_wrapper,
        return_value_policy<return_opaque_pointer>());
    def("stream_synchronize", &cudaStreamSynchronize_wrapper);

    // reset scope
    scope();
}
