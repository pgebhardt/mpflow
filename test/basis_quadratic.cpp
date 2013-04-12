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
                std::make_tuple(0.60383564, 0.79710889),
                std::make_tuple(0.27702361, -0.06758346),
                std::make_tuple(0.99972898, 0.02327905),
                std::make_tuple(0.44042963, 0.36476272),
                std::make_tuple(0.6383763, -0.02215221),
                std::make_tuple(0.80178231, 0.41019398)
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
            std::make_tuple(0.60383564, 0.79710889),
            std::make_tuple(0.27702361, -0.06758346),
            std::make_tuple(0.99972898, 0.02327905),
            std::make_tuple(0.44042963, 0.36476272),
            std::make_tuple(0.6383763, -0.02215221),
            std::make_tuple(0.80178231, 0.41019398)
        }};

    // create basis function
    std::shared_ptr<fastEIT::basis::Quadratic> basis;
    EXPECT_NO_THROW({
        basis = std::make_shared<fastEIT::basis::Quadratic>(nodes, 0);
    });

    // check coefficients
    EXPECT_NEAR(basis->coefficients()[0], -0.09342247, 1e-5);
    EXPECT_NEAR(basis->coefficients()[1], 0.0767253, 1e-5);
    EXPECT_NEAR(basis->coefficients()[2], -0.610261, 1e-5);
    EXPECT_NEAR(basis->coefficients()[3], 0.0466059, 1e-5);
    EXPECT_NEAR(basis->coefficients()[4], 2.9484474, 1e-5);
    EXPECT_NEAR(basis->coefficients()[5], -0.74139021, 1e-5);

    // check error
    EXPECT_THROW(
        std::make_shared<fastEIT::basis::Quadratic>(nodes, 7),
        std::invalid_argument);
};

// basis function definition
TEST_F(BasisQuadraticTest, Definition) {
    // check basis function definition
    for (fastEIT::dtype::index basis = 0; basis < fastEIT::basis::Quadratic::nodes_per_element;
        ++basis)
    for (fastEIT::dtype::index node = 0; node < fastEIT::basis::Quadratic::nodes_per_element;
        ++node) {
        if (basis == node) {
            EXPECT_NEAR(basis_[basis]->evaluate(basis_[basis]->nodes()[node]), 1.0, 1e-6);
        } else {
            EXPECT_NEAR(basis_[basis]->evaluate(basis_[basis]->nodes()[node]), 0.0, 1e-6);
        }
    }
};
