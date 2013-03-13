#ifndef PYFASTEIT_CUDA_HPP
#define PYFASTEIT_CUDA_HPP

namespace pyfasteit {
    void export_cuda();
}

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(CUstream_st)
BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(cublasContext)

#endif
