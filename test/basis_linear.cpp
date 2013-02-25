#include <cmath>
#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

// test class
class BasisLinearTest :
    public ::testing::Test {
protected:
    void SetUp() {
        // nodes array
        std::array<std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>,
            fastEIT::basis::Linear::nodes_per_element> nodes = {{
                std::make_tuple(0.0f, 0.0f),
                std::make_tuple(1.0f, 0.0f),
                std::make_tuple(0.0f, 1.0f)
            }};

        // create basis function
        for (fastEIT::dtype::index node = 0;
            node < fastEIT::basis::Linear::nodes_per_element;
            ++node) {
            this->basis_[node] = std::make_shared<fastEIT::basis::Linear>(nodes, node);
        }
    }

    std::array<std::shared_ptr<fastEIT::basis::Linear>,
        fastEIT::basis::Linear::nodes_per_element> basis_;
};

// constructor test
TEST_F(BasisLinearTest, Constructor) {
    // nodes array
    std::array<std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>,
        fastEIT::basis::Linear::nodes_per_element> nodes = {{
            std::make_tuple(1.0f, 0.0f),
            std::make_tuple(0.0f, 1.0f),
            std::make_tuple(0.0f, 0.0f)
        }};

    // create basis function
    std::shared_ptr<fastEIT::basis::Linear> basis;
    EXPECT_NO_THROW({
        basis = std::make_shared<fastEIT::basis::Linear>(nodes, 0);
    });

    // check member
    EXPECT_EQ(basis->coefficients()[0], 0.0f);
    EXPECT_EQ(basis->coefficients()[1], 1.0f);
    EXPECT_EQ(basis->coefficients()[2], 0.0f);
    EXPECT_EQ(basis->nodes()[0], std::make_tuple(1.0f, 0.0f));
    EXPECT_EQ(basis->nodes()[1], std::make_tuple(0.0f, 1.0f));
    EXPECT_EQ(basis->nodes()[2], std::make_tuple(0.0f, 0.0f));

    // check error
    EXPECT_THROW(
        std::make_shared<fastEIT::basis::Linear>(nodes, 4),
        std::invalid_argument);
};

// basis function definition
TEST_F(BasisLinearTest, Definition) {
    // check basis function definition
    for (fastEIT::dtype::index basis = 0; basis < 3; ++basis)
    for (fastEIT::dtype::index node = 0; node < 3; ++node) {
        if (basis == node) {
            EXPECT_EQ(basis_[basis]->evaluate(basis_[basis]->nodes()[node]), 1.0f);
        } else {
            EXPECT_EQ(basis_[basis]->evaluate(basis_[basis]->nodes()[node]), 0.0f);
        }
    }
};

// integrate with basis
TEST_F(BasisLinearTest, IntegrateWithBasis) {
    // solution computed by sympy
    auto solution = [](std::shared_ptr<fastEIT::basis::Linear> self,
        std::shared_ptr<fastEIT::basis::Linear> other) -> fastEIT::dtype::real {
        return 1.0 * ((((((((((((((((((((((((((((((((((((((((((self->coefficients()[0] * other->coefficients()[0] / 2.0 + self->coefficients()[0] * other->coefficients()[1] * std::get<0>(self->nodes()[0]) / 6.0) + self->coefficients()[0] * other->coefficients()[1] * std::get<0>(self->nodes()[1]) / 6.0) + self->coefficients()[0] * other->coefficients()[1] * std::get<0>(self->nodes()[2]) / 6.0) + self->coefficients()[0] * other->coefficients()[2] * std::get<1>(self->nodes()[0]) / 6.0) + self->coefficients()[0] * other->coefficients()[2] * std::get<1>(self->nodes()[1]) / 6.0) + self->coefficients()[0] * other->coefficients()[2] * std::get<1>(self->nodes()[2]) / 6.0) + other->coefficients()[0] * self->coefficients()[1] * std::get<0>(self->nodes()[0]) / 6.0) + other->coefficients()[0] * self->coefficients()[1] * std::get<0>(self->nodes()[1]) / 6.0) + other->coefficients()[0] * self->coefficients()[1] * std::get<0>(self->nodes()[2]) / 6.0) + other->coefficients()[0] * self->coefficients()[2] * std::get<1>(self->nodes()[0]) / 6.0) + other->coefficients()[0] * self->coefficients()[2] * std::get<1>(self->nodes()[1]) / 6.0) + other->coefficients()[0] * self->coefficients()[2] * std::get<1>(self->nodes()[2]) / 6.0) + self->coefficients()[1] * other->coefficients()[1] * std::get<0>(self->nodes()[0]) * std::get<0>(self->nodes()[0]) / 12.0) + self->coefficients()[1] * other->coefficients()[1] * std::get<0>(self->nodes()[0]) * std::get<0>(self->nodes()[1]) / 12.0) + self->coefficients()[1] * other->coefficients()[1] * std::get<0>(self->nodes()[0]) * std::get<0>(self->nodes()[2]) / 12.0) + self->coefficients()[1] * other->coefficients()[1] * std::get<0>(self->nodes()[1]) * std::get<0>(self->nodes()[1]) / 12.0) + self->coefficients()[1] * other->coefficients()[1] * std::get<0>(self->nodes()[1]) * std::get<0>(self->nodes()[2]) / 12.0) + self->coefficients()[1] * other->coefficients()[1] * std::get<0>(self->nodes()[2]) * std::get<0>(self->nodes()[2]) / 12.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[0]) * std::get<1>(self->nodes()[0]) / 12.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[0]) * std::get<1>(self->nodes()[1]) / 24.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[0]) * std::get<1>(self->nodes()[2]) / 24.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[1]) * std::get<1>(self->nodes()[0]) / 24.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[1]) * std::get<1>(self->nodes()[1]) / 12.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[1]) * std::get<1>(self->nodes()[2]) / 24.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[2]) * std::get<1>(self->nodes()[0]) / 24.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[2]) * std::get<1>(self->nodes()[1]) / 24.0) + self->coefficients()[1] * other->coefficients()[2] * std::get<0>(self->nodes()[2]) * std::get<1>(self->nodes()[2]) / 12.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[0]) * std::get<1>(self->nodes()[0]) / 12.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[0]) * std::get<1>(self->nodes()[1]) / 24.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[0]) * std::get<1>(self->nodes()[2]) / 24.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[1]) * std::get<1>(self->nodes()[0]) / 24.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[1]) * std::get<1>(self->nodes()[1]) / 12.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[1]) * std::get<1>(self->nodes()[2]) / 24.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[2]) * std::get<1>(self->nodes()[0]) / 24.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[2]) * std::get<1>(self->nodes()[1]) / 24.0) + other->coefficients()[1] * self->coefficients()[2] * std::get<0>(self->nodes()[2]) * std::get<1>(self->nodes()[2]) / 12.0) + self->coefficients()[2] * other->coefficients()[2] * std::get<1>(self->nodes()[0]) * std::get<1>(self->nodes()[0]) / 12.0) + self->coefficients()[2] * other->coefficients()[2] * std::get<1>(self->nodes()[0]) * std::get<1>(self->nodes()[1]) / 12.0) + self->coefficients()[2] * other->coefficients()[2] * std::get<1>(self->nodes()[0]) * std::get<1>(self->nodes()[2]) / 12.0) + self->coefficients()[2] * other->coefficients()[2] * std::get<1>(self->nodes()[1]) * std::get<1>(self->nodes()[1]) / 12.0) + self->coefficients()[2] * other->coefficients()[2] * std::get<1>(self->nodes()[1]) * std::get<1>(self->nodes()[2]) / 12.0) + self->coefficients()[2] * other->coefficients()[2] * std::get<1>(self->nodes()[2]) * std::get<1>(self->nodes()[2]) / 12.0) * std::abs(((-std::get<0>(self->nodes()[0]) + std::get<0>(self->nodes()[1])) * (-std::get<1>(self->nodes()[0]) + std::get<1>(self->nodes()[2])) - (-std::get<0>(self->nodes()[0]) + std::get<0>(self->nodes()[2])) * (-std::get<1>(self->nodes()[0]) + std::get<1>(self->nodes()[1]))));
    };

    // check all permutations
    for (fastEIT::dtype::index i = 0; i < 3; ++i)
    for (fastEIT::dtype::index j = 0; j < 3; ++j) {
        EXPECT_LT(std::abs(basis_[i]->integrateWithBasis(basis_[j]) -
            solution(basis_[i], basis_[j])), 1e-6);
    }
};

// integrate gradient with basis
TEST_F(BasisLinearTest, IntegrateGradientWithBasis) {
    // solution computed by sympy
    auto solution = [](std::shared_ptr<fastEIT::basis::Linear> self,
        std::shared_ptr<fastEIT::basis::Linear> other) -> fastEIT::dtype::real {
        return 1.0 * (self->coefficients()[1] * other->coefficients()[1] / 2.0 + self->coefficients()[2] * other->coefficients()[2] / 2.0) * std::abs(((-std::get<0>(self->nodes()[0]) + std::get<0>(self->nodes()[1])) * (-std::get<1>(self->nodes()[0]) + std::get<1>(self->nodes()[2])) - (-std::get<0>(self->nodes()[0]) + std::get<0>(self->nodes()[2])) * (-std::get<1>(self->nodes()[0]) + std::get<1>(self->nodes()[1]))));
    };

    // check all permutations
    for (fastEIT::dtype::index i = 0; i < 3; ++i)
    for (fastEIT::dtype::index j = 0; j < 3; ++j) {
        EXPECT_LT(std::abs(basis_[i]->integrateGradientWithBasis(basis_[j]) -
            solution(basis_[i], basis_[j])), 1e-6);
    }
};
