#include <random>
#include "gtest/gtest.h"
#include "../include/fasteit.h"

// check matrix for equality
int matrixCompare(std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> A,
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> B) {
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

TEST(MatrixTest, Add) {
    // init random generator
    std::default_random_engine generator;
    std::uniform_real_distribution<fastEIT::dtype::real> distribution(-10.0, 10.0);

    // create matrices
    auto A = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 32, nullptr);
    auto B = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 32, nullptr);

    // fill A and B with random numbers
    for (fastEIT::dtype::index row = 0; row < A->rows(); ++row) {
        for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
            (*A)(row, column) = distribution(generator);
            (*B)(row, column) = distribution(generator);
        }
    }

    // upload to device
    A->copyToDevice(nullptr);
    B->copyToDevice(nullptr);

    // add on device
    A->add(B, nullptr);

    // add on CPU
    for (fastEIT::dtype::index row = 0; row < A->rows(); ++row) {
        for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
             (*B)(row, column) += (*A)(row, column);
        }
    }

    // copy to host
    A->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(matrixCompare(A, B), 0);
}
