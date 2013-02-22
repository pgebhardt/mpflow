#include <random>
#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

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

    // member
    cublasHandle_t handle_;
};

TEST_F(MatrixTest, Constructor) {
    // create matrices with invalid size
    EXPECT_THROW(
        std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 0, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(0, 32, nullptr),
        std::invalid_argument);

    // create correct matrix
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> A = nullptr;
    EXPECT_NO_THROW({
        A = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 12, nullptr);
    });

    // check member
    EXPECT_EQ(A->rows(), 10U);
    EXPECT_EQ(A->columns(), 12U);
};

TEST_F(MatrixTest, Access) {
    // create matrix
    auto A = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 12, nullptr);

    // check access
    EXPECT_NO_THROW((*A)(9, 8));
    EXPECT_NO_THROW((*A)(3, 7));
    EXPECT_THROW((*A)(10, 7), std::logic_error);
    EXPECT_THROW((*A)(1, 17), std::logic_error);
};

TEST_F(MatrixTest, Copy) {
    // create matrices
    auto A = randomMatrix(32, 48, nullptr);
    auto B = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 48, nullptr);
    auto C = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(48, 32, nullptr);

    // copy
    EXPECT_NO_THROW(A->copyToDevice(nullptr));
    EXPECT_NO_THROW(B->copy(A, nullptr));
    EXPECT_NO_THROW(B->copyToHost(nullptr));
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(matrixCompare(A, B), 0);

    // expect error
    EXPECT_THROW(
        C->copy(A, nullptr),
        std::logic_error);
};

TEST_F(MatrixTest, Add) {
    // create matrices
    auto A = randomMatrix(32, 32, nullptr);
    auto B = randomMatrix(32, 32, nullptr);
    auto C = randomMatrix(33, 32, nullptr);

    // add on device
    EXPECT_NO_THROW(A->add(B, nullptr));

    // add on CPU
    for (fastEIT::dtype::index row = 0; row < A->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        (*B)(row, column) += (*A)(row, column);
    }

    // copy to host
    A->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(matrixCompare(A, B), 0);

    // expect error
    EXPECT_THROW(
        A->add(C, nullptr),
        std::logic_error);
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

    // expect error
    EXPECT_THROW(
        C_GPU->multiply(B, A, handle_, nullptr),
        std::logic_error);
};

TEST_F(MatrixTest, ScalarMultiply) {
    // create matrices
    auto A = randomMatrix(32, 48, nullptr);
    auto B = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 48, nullptr);

    // scalar multiply
    EXPECT_NO_THROW(A->scalarMultiply(3.0, nullptr));
    B->copy(A, nullptr);

    // scalar multiply on CPU
    for (fastEIT::dtype::index row = 0; row < A->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        (*A)(row, column) *= 3.0;
    }

    // copy to host
    B->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare
    EXPECT_EQ(matrixCompare(A, B), 0);
};

TEST_F(MatrixTest, VectorDotProduct) {
    // create matrices
    auto A = randomMatrix(32, 16, nullptr);
    auto B = randomMatrix(32, 16, nullptr);
    auto C_GPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 16, nullptr);
    auto C_CPU = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 16, nullptr);
    auto D = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(16, 16, nullptr);

    // vector dot product
    EXPECT_NO_THROW(C_GPU->vectorDotProduct(A, B, nullptr));

    // vector dot product on CPU
    for (fastEIT::dtype::index row = 0; row < A->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        (*C_CPU)(0, column) += (*A)(row, column) * (*B)(row, column);
    }

    // copy to host
    C_GPU->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare first row
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        EXPECT_LT(std::abs((*C_GPU)(0, column) - (*C_CPU)(0, column)), 1e-3);
    }

    // expect error
    EXPECT_THROW(
        D->vectorDotProduct(A, B, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        A->vectorDotProduct(D, B, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        B->vectorDotProduct(D, A, nullptr),
        std::invalid_argument);
}

TEST_F(MatrixTest, Sum) {
    // create matrices
    auto A = randomMatrix(32, 16, nullptr);
    auto B = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 16, nullptr);
    auto C = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(16, 16, nullptr);

    // sum
    EXPECT_NO_THROW(B->sum(A, nullptr));

    // sum on CPU
    for (fastEIT::dtype::index row = 1; row < A->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        (*A)(0, column) += (*A)(row, column);
    }

    // copy to host
    B->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare first row
    for (fastEIT::dtype::index column = 0; column < A->columns(); ++column) {
        EXPECT_LT(std::abs((*B)(0, column) - (*A)(0, column)), 1e-3);
    }

    // expect error
    EXPECT_THROW(
        C->sum(A, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        A->sum(C, nullptr),
        std::invalid_argument);
};

TEST_F(MatrixTest, Min) {
    // create matrices
    auto A = randomMatrix(32, 1, nullptr);
    auto B = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 1, nullptr);
    auto C = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(16, 1, nullptr);

    // sum
    EXPECT_NO_THROW(B->min(A, nullptr));

    // sum on CPU
    for (fastEIT::dtype::index row = 1; row < A->rows(); ++row) {
        (*A)(0, 0) = std::min((*A)(0, 0), (*A)(row, 0));
    }

    // copy to host
    B->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare first element
    EXPECT_EQ((*B)(0, 0), (*A)(0, 0));

    // expect error
    EXPECT_THROW(
        C->min(A, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        A->min(C, nullptr),
        std::invalid_argument);
};

TEST_F(MatrixTest, Max) {
    // create matrices
    auto A = randomMatrix(32, 1, nullptr);
    auto B = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(32, 1, nullptr);
    auto C = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(16, 1, nullptr);

    // sum
    EXPECT_NO_THROW(B->max(A, nullptr));

    // sum on CPU
    for (fastEIT::dtype::index row = 1; row < A->rows(); ++row) {
        (*A)(0, 0) = std::max((*A)(0, 0), (*A)(row, 0));
    }

    // copy to host
    B->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // compare first element
    EXPECT_EQ((*B)(0, 0), (*A)(0, 0));

    // expect error
    EXPECT_THROW(
        C->max(A, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        A->max(C, nullptr),
        std::invalid_argument);
};

TEST_F(MatrixTest, LoadSaveTxt) {
    // create matrix
    auto A = randomMatrix(32, 32, nullptr);

    // create stringstream
    std::stringstream sstream;

    // save matrix to stream
    EXPECT_NO_THROW(fastEIT::matrix::savetxt(A, &sstream));

    // load matrix from stream
    std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> B = nullptr;
    EXPECT_NO_THROW({
        B = fastEIT::matrix::loadtxt<fastEIT::dtype::real>(&sstream, nullptr);
    });

    // compare matrices
    EXPECT_EQ(matrixCompare(A, B), 0);
};
