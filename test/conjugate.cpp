#include <cmath>
#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

// test class
class ConjugateTest :
    public ::testing::Test {
protected:
    void SetUp() {
        cublasCreate(&this->handle_);
    }
    void TearDown() {
        cublasDestroy(this->handle_);
    }

    cublasHandle_t handle_;
};

TEST_F(ConjugateTest, Constructor) {
    // create solver
    std::shared_ptr<fastEIT::numeric::Conjugate> conjugate = nullptr;
    EXPECT_NO_THROW({
        conjugate = std::make_shared<fastEIT::numeric::Conjugate>(20, handle_, nullptr);
    });

    // check errors
    EXPECT_THROW(
        std::make_shared<fastEIT::numeric::Conjugate>(0, handle_, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::numeric::Conjugate>(7, nullptr, nullptr),
        std::invalid_argument);
};

TEST_F(ConjugateTest, Solve) {
    // create solver
    auto solver = std::make_shared<fastEIT::numeric::Conjugate>(4, handle_, nullptr);

    // init matrix
    auto x = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 1, nullptr);
    auto A = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 4, nullptr);
    std::tie((*A)(0, 0), (*A)(0, 1), (*A)(0, 2), (*A)(0, 3)) =
        std::make_tuple(1.0f, 2.0f, 0.0f, 0.0f);
    std::tie((*A)(1, 0), (*A)(1, 1), (*A)(1, 2), (*A)(1, 3)) =
        std::make_tuple(2.0f, 1.0f, 2.0f, 0.0f);
    std::tie((*A)(2, 0), (*A)(2, 1), (*A)(2, 2), (*A)(2, 3)) =
        std::make_tuple(0.0f, 2.0f, 1.0f, 2.0f);
    std::tie((*A)(3, 0), (*A)(3, 1), (*A)(3, 2), (*A)(3, 3)) =
        std::make_tuple(0.0f, 0.0f, 2.0f, 1.0f);
    std::tie((*x)(0, 0), (*x)(1, 0), (*x)(2, 0), (*x)(3, 0)) =
        std::make_tuple(0.8f, -0.4f, -0.6f, 1.2f);

    A->copyToDevice(nullptr);

    // excitation
    auto f = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 1, nullptr);
    (*f)(2, 0) = 1.0f;
    f->copyToDevice(nullptr);

    // solve
    auto x0 = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 1, nullptr);
    EXPECT_NO_THROW(solver->solve(A, f, 4, handle_, nullptr, x0));

    x0->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // check result
    for (fastEIT::dtype::real row = 0; row < x->rows(); ++row) {
        EXPECT_LT(std::abs((*x)(row, 0) - (*x0)(row, 0)), 1e-6f);
    }

    // check error
    EXPECT_THROW(
        solver->solve(nullptr, f, 4, handle_, nullptr, x),
        std::invalid_argument);
    EXPECT_THROW(
        solver->solve(A, nullptr, 4, handle_, nullptr, x),
        std::invalid_argument);
    EXPECT_THROW(
        solver->solve(A, f, 4, nullptr, nullptr, x),
        std::invalid_argument);
    EXPECT_THROW(
        solver->solve(A, f, 4, handle_, nullptr, nullptr),
        std::invalid_argument);
};
