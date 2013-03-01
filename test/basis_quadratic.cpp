#include <cmath>
#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

// test class
class BasisQuadraticTest :
    public ::testing::Test {
protected:
    void SetUp() {
        // nodes array
        std::array<std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>,
            fastEIT::basis::Quadratic::nodes_per_element> nodes = {{
                std::make_tuple(0.0f, 0.0f),
                std::make_tuple(1.0f, 0.0f),
                std::make_tuple(0.0f, 1.0f),
                std::make_tuple(0.5f, 0.0f),
                std::make_tuple(0.5f, 0.5f),
                std::make_tuple(0.0f, 0.5f)
            }};

        // create basis function
        for (fastEIT::dtype::index node = 0;
            node < fastEIT::basis::Quadratic::nodes_per_element;
            ++node) {
            this->basis_[node] = std::make_shared<fastEIT::basis::Quadratic>(nodes, node);
        }
    }

    std::array<std::shared_ptr<fastEIT::basis::Quadratic>,
        fastEIT::basis::Quadratic::nodes_per_element> basis_;
};

// constructor test
TEST_F(BasisQuadraticTest, Constructor) {
    // nodes array
    std::array<std::tuple<fastEIT::dtype::real, fastEIT::dtype::real>,
        fastEIT::basis::Quadratic::nodes_per_element> nodes = {{
            std::make_tuple(0.0f, 0.0f),
            std::make_tuple(1.0f, 0.0f),
            std::make_tuple(0.0f, 1.0f),
            std::make_tuple(0.5f, 0.0f),
            std::make_tuple(0.5f, 0.5f),
            std::make_tuple(0.0f, 0.5f)
        }};

    // create basis function
    std::shared_ptr<fastEIT::basis::Quadratic> basis;
    EXPECT_NO_THROW({
        basis = std::make_shared<fastEIT::basis::Quadratic>(nodes, 0);
    });

    // check member
    EXPECT_FLOAT_EQ(basis->coefficients()[0], 1.0f);
    EXPECT_FLOAT_EQ(basis->coefficients()[1], -3.0f);
    EXPECT_FLOAT_EQ(basis->coefficients()[2], -3.0f);
    EXPECT_FLOAT_EQ(basis->coefficients()[3], 2.0f);
    EXPECT_FLOAT_EQ(basis->coefficients()[4], 2.0f);
    EXPECT_FLOAT_EQ(basis->coefficients()[5], 4.0f);
    EXPECT_EQ(basis->nodes()[0], std::make_tuple(0.0f, 0.0f));
    EXPECT_EQ(basis->nodes()[1], std::make_tuple(1.0f, 0.0f));
    EXPECT_EQ(basis->nodes()[2], std::make_tuple(0.0f, 1.0f));
    EXPECT_EQ(basis->nodes()[3], std::make_tuple(0.5f, 0.0f));
    EXPECT_EQ(basis->nodes()[4], std::make_tuple(0.5f, 0.5f));
    EXPECT_EQ(basis->nodes()[5], std::make_tuple(0.0f, 0.5f));

    // check error
    EXPECT_THROW(
        std::make_shared<fastEIT::basis::Quadratic>(nodes, 7),
        std::invalid_argument);
};

// basis function definition
TEST_F(BasisQuadraticTest, Definition) {
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
