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
                std::make_tuple(0.60383564, 0.79710889),
                std::make_tuple(0.27702361, -0.06758346),
                std::make_tuple(0.99972898, 0.02327905)
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
            std::make_tuple(0.60383564, 0.79710889),
            std::make_tuple(0.27702361, -0.06758346),
            std::make_tuple(0.99972898, 0.02327905)
        }};

    // create basis function
    std::shared_ptr<fastEIT::basis::Linear> basis;
    EXPECT_NO_THROW({
        basis = std::make_shared<fastEIT::basis::Linear>(nodes, 0);
    });

    // check member
    EXPECT_FLOAT_EQ(basis->coefficients()[0], 0.12434669);
    EXPECT_FLOAT_EQ(basis->coefficients()[1], -0.15265293);
    EXPECT_FLOAT_EQ(basis->coefficients()[2], 1.21417613);

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
            EXPECT_LE(std::abs(basis_[basis]->evaluate(basis_[basis]->nodes()[node]) - 1.0f), 1e-6);
        } else {
            EXPECT_LE(std::abs(basis_[basis]->evaluate(basis_[basis]->nodes()[node])), 1e-6);
        }
    }
};
