#include <cmath>
#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

TEST(SparseConjugateTest, Constructor) {
    // create solver
    std::shared_ptr<fastEIT::numeric::SparseConjugate> conjugate = nullptr;
    EXPECT_NO_THROW({
        conjugate = std::make_shared<fastEIT::numeric::SparseConjugate>(20, 2, nullptr);
    });

    // check errors
    EXPECT_THROW(
        std::make_shared<fastEIT::numeric::SparseConjugate>(0, 2, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::numeric::SparseConjugate>(3, 0, nullptr),
        std::invalid_argument);
};

TEST(SparseConjugateTest, Solve) {
    // create solver
    auto solver = std::make_shared<fastEIT::numeric::SparseConjugate>(4, 4, nullptr);

    // init matrix
    auto x = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 4, nullptr);
    auto A_dense = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 4, nullptr);
    std::tie((*A_dense)(0, 0), (*A_dense)(0, 1), (*A_dense)(0, 2), (*A_dense)(0, 3)) =
        std::make_tuple(2.0f, 1.0f, 0.0f, 0.0f);
    std::tie((*A_dense)(1, 0), (*A_dense)(1, 1), (*A_dense)(1, 2), (*A_dense)(1, 3)) =
        std::make_tuple(1.0f, 2.0f, 1.0f, 0.0f);
    std::tie((*A_dense)(2, 0), (*A_dense)(2, 1), (*A_dense)(2, 2), (*A_dense)(2, 3)) =
        std::make_tuple(0.0f, 1.0f, 2.0f, 1.0f);
    std::tie((*A_dense)(3, 0), (*A_dense)(3, 1), (*A_dense)(3, 2), (*A_dense)(3, 3)) =
        std::make_tuple(0.0f, 0.0f, 1.0f, 2.0f);

    // init solution
    std::tie((*x)(0, 0), (*x)(0, 1), (*x)(0, 2), (*x)(0, 3)) =
        std::make_tuple(0.8f, -0.6f, 0.4f, -0.2f);
    std::tie((*x)(1, 0), (*x)(1, 1), (*x)(1, 2), (*x)(1, 3)) =
        std::make_tuple(-0.6f, 1.2f, -0.8f, 0.4f);
    std::tie((*x)(2, 0), (*x)(2, 1), (*x)(2, 2), (*x)(2, 3)) =
        std::make_tuple(0.4f, -0.8f, 1.2f, -0.6f);
    std::tie((*x)(3, 0), (*x)(3, 1), (*x)(3, 2), (*x)(3, 3)) =
        std::make_tuple(-0.2f, 0.4f, -0.6f, 0.8f);

    A_dense->copyToDevice(nullptr);
    auto A = std::make_shared<fastEIT::SparseMatrix<fastEIT::dtype::real>>(A_dense, nullptr);

    // excitation
    auto f = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 4, nullptr);
    (*f)(0, 0) = 1.0f;
    (*f)(1, 1) = 1.0f;
    (*f)(2, 2) = 1.0f;
    (*f)(3, 3) = 1.0f;
    f->copyToDevice(nullptr);

    // solve
    auto x0 = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 4, nullptr);
    EXPECT_NO_THROW(solver->solve(A, f, 4, false, nullptr, x0));

    x0->copyToHost(nullptr);
    cudaStreamSynchronize(nullptr);

    // check result
    for (fastEIT::dtype::real row = 0; row < x->rows(); ++row)
    for (fastEIT::dtype::real column = 0; column < x->columns(); ++column) {
        EXPECT_LT(std::fabs((*x)(row, column) - (*x0)(row, column)), 1e-6f);
    }

    // check error
    EXPECT_THROW(
        solver->solve(nullptr, f, 4, false, nullptr, x),
        std::invalid_argument);
    EXPECT_THROW(
        solver->solve(A, nullptr, 4, false, nullptr, x),
        std::invalid_argument);
    EXPECT_THROW(
        solver->solve(A, f, 4, false, nullptr, nullptr),
        std::invalid_argument);
};
