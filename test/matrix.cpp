#include <random>
#include "gtest/gtest.h"
#include "../include/fasteit.h"

// test class
class MatrixTest :
    public ::testing::Test {
protected:
    void SetUp() {
        cublasCreate(&this->handle_);
    }

    void TearDown() {
        cublasDestroy(this->handle_);
    }

    // generate random real matrix
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> randomMatrix(
        fastEIT::dtype::size rows, fastEIT::dtype::size columns,
        cudaStream_t stream) {
        // init random generator
        std::default_random_engine generator;
        std::uniform_real_distribution<fastEIT::dtype::real> distribution(-10.0, 10.0);

        // create matrices
        auto matrix = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(
            rows, columns, stream);

        // fill matrix with random numbers
        for (fastEIT::dtype::index row = 0; row < matrix->rows(); ++row) {
            for (fastEIT::dtype::index column = 0; column < matrix->columns(); ++column) {
                (*matrix)(row, column) = distribution(generator);
            }
        }

        // upload to device
        matrix->copyToDevice(nullptr);

        return matrix;
    }

    // compare matrices on CPU
    template <
        class type
    >
    int matrixCompare(std::shared_ptr<fastEIT::Matrix<type>> A,
        std::shared_ptr<fastEIT::Matrix<type>> B) {
        // check sizes
        if ((A->rows() != B->rows()) || (A->columns() != B->columns())) {
            throw std::invalid_argument("matrixCompare: matrices not of same size");
        }

        // compare matrices
        int result = 0;
        for (fastEIT::dtype::index row = 0; row < A->rows(); ++row) {
            for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
                result += (*B)(row, column) - (*A)(row, column);
            }
        }

        return result;
    }

    // member
    cublasHandle_t handle_;
};

TEST_F(MatrixTest, Add) {
    // create matrices
    auto A = randomMatrix(32, 32, nullptr);
    auto B = randomMatrix(32, 32, nullptr);

    // add on device
    A->add(B, nullptr);

    // add on CPU
    for (fastEIT::dtype::index row = 0; row < A->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        (*B)(row, column) += (*A)(row, column);
    }

    // copy to host
    A->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(this->matrixCompare(A, B), 0);
};

TEST_F(MatrixTest, Multiply) {
    // create matrices
    auto A = randomMatrix(32, 64, nullptr);
    auto B = randomMatrix(64, 32, nullptr);
    auto C_GPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 32, nullptr);
    auto C_CPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 32, nullptr);

    // multiply on device
    EXPECT_NO_THROW(C_GPU->multiply(A, B, handle_, nullptr));

    // multiply on CPU
    for (fastEIT::dtype::index row = 0; row < C_CPU->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < C_CPU->columns(); ++column)
    for (fastEIT::dtype::index i = 0; i < A->columns(); ++i) {
        (*C_CPU)(row, column) += (*A)(row, i) * (*B)(i, column);
    }

    // copy to host
    C_GPU->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(this->matrixCompare(C_GPU, C_CPU), 0);
};
