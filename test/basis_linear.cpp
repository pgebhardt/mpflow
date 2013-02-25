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
