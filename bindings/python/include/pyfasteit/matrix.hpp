#ifndef PYFASTEIT_MATRIX_HPP
#define PYFASTEIT_MATRIX_HPP

namespace pyfasteit {
    void export_matrix();

    template <
        class matrix_type,
        int npy_type
    >
    std::shared_ptr<fastEIT::Matrix<matrix_type>> fromNumpy(boost::python::numeric::array& array,
        cudaStream_t stream);
}

#endif
