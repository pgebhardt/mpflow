#include <random>
#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

// test class
class SparseMatrixTest :
    public ::testing::Test {
protected:
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
        for (fastEIT::dtype::index row = 0; row < matrix->rows(); ++row)
        for (fastEIT::dtype::index column = 0; column < matrix->columns(); ++column) {
            (*matrix)(row, column) = distribution(generator);
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
        for (fastEIT::dtype::index row = 0; row < A->rows(); ++row)
        for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
            result += (*B)(row, column) - (*A)(row, column);
        }

        return result;
    }
};

TEST_F(SparseMatrixTest, Constructor) {
    // create matrices
    auto matrix = randomMatrix(30, 10, nullptr);

    // create sparse matrix
    std::shared_ptr<fastEIT::SparseMatrix<fastEIT::dtype::real>> sparse_matrix = nullptr;
    EXPECT_NO_THROW({
        sparse_matrix = std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(matrix, nullptr);
    });

    // check member
    EXPECT_EQ(sparse_matrix->rows(), matrix->rows());
    EXPECT_EQ(sparse_matrix->columns(), matrix->columns());
    EXPECT_EQ(sparse_matrix->density(), matrix->columns());

    // check error
    EXPECT_THROW(
        std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(nullptr, nullptr),
        std::invalid_argument);
}

TEST_F(SparseMatrixTest, Convert) {
    // create dense matrix
    auto dense = randomMatrix(10, 8, nullptr);

    // create sparse matrix
    auto sparse_matrix = std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(dense, nullptr);

    // convert to matrix and compare
    auto convert = sparse_matrix->toMatrix(nullptr);

    // copy to host
    convert->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    EXPECT_EQ(matrixCompare(dense, convert), 0);

    // density over block size
    dense = randomMatrix(10, fastEIT::sparseMatrix::block_size * 2, nullptr);

    // convert to sparse matrix
    sparse_matrix = std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(dense, nullptr);

    // convert to matrix and compare
    convert = sparse_matrix->toMatrix(nullptr);

    // copy to host
    convert->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    EXPECT_NE(matrixCompare(dense, convert), 0);
}

TEST_F(SparseMatrixTest, MatrixVectorMultiply) {
    // create matrices
    auto matrix = randomMatrix(20, 10, nullptr);
    auto vector = randomMatrix(10, 1, nullptr);

    // create sparse matrix
    auto sparse_matrix = std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(matrix, nullptr);

    // multiply
    auto result_GPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(20, 1, nullptr);
    EXPECT_NO_THROW(sparse_matrix->multiply(vector, nullptr, result_GPU));

    // multiply on CPU
    auto result_CPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(20, 1, nullptr);
    for (fastEIT::dtype::index row = 0; row < result_CPU->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < result_CPU->columns(); ++column)
    for (fastEIT::dtype::index i = 0; i < matrix->columns(); ++i) {
        (*result_CPU)(row, column) += (*matrix)(row, i) * (*vector)(i, column);
    }

    // copy result to host
    result_GPU->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(matrixCompare(result_GPU, result_CPU), 0);

    // multiply with wrong size
    EXPECT_THROW(
        sparse_matrix->multiply(vector, nullptr,
            std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 2, nullptr)),
        std::invalid_argument);
    EXPECT_THROW(
        sparse_matrix->multiply(
            std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 2, nullptr),
            nullptr, result_GPU),
        std::invalid_argument);
};

TEST_F(SparseMatrixTest, MatrixMatrixMultiply) {
    // create matrices
    auto matrix = randomMatrix(20, 10, nullptr);
    auto vector = randomMatrix(10, 4, nullptr);

    // create sparse matrix
    auto sparse_matrix = std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(matrix, nullptr);

    // multiply
    auto result_GPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(20, 4, nullptr);
    EXPECT_NO_THROW(sparse_matrix->multiply(vector, nullptr, result_GPU));

    // multiply on CPU
    auto result_CPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(20, 4, nullptr);
    for (fastEIT::dtype::index row = 0; row < result_CPU->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < result_CPU->columns(); ++column)
    for (fastEIT::dtype::index i = 0; i < matrix->columns(); ++i) {
        (*result_CPU)(row, column) += (*matrix)(row, i) * (*vector)(i, column);
    }

    // copy result to host
    result_GPU->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(matrixCompare(result_GPU, result_CPU), 0);

    // multiply with wrong size
    EXPECT_THROW(
        sparse_matrix->multiply(vector, nullptr,
            std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 1, nullptr)),
        std::invalid_argument);
    EXPECT_THROW(
        sparse_matrix->multiply(
            std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(1, 40, nullptr),
            nullptr, result_GPU),
        std::invalid_argument);
};
