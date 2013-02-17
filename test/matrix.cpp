#include <random>
#include "gtest/gtest.h"
#include "../include/fasteit.h"

// test class
class MatrixTest :
    public ::testing::Test {
protected:
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

    // add matrices on CPU
    template <
        class type
    >
    void add(std::shared_ptr<fastEIT::Matrix<type>> A,
        std::shared_ptr<fastEIT::Matrix<type>> B) {
        // check sizes
        if ((A->rows() != B->rows()) || (A->columns() != B->columns())) {
            throw std::invalid_argument("MatrixTest::add: matrices not of same size");
        }

        // add matrices
        for (fastEIT::dtype::index row = 0; row < A->rows(); ++row) {
            for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
                (*A)(row, column) += (*B)(row, column);
            }
        }
    }
};

TEST_F(MatrixTest, Add) {
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
    this->add(B, A);

    // copy to host
    A->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(this->matrixCompare(A, B), 0);
};
