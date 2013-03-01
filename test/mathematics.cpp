#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

TEST(MathematicsTest, GaussElemination) {
    // create matrix
    std::array<std::array<float, 4>, 4> matrix;
    matrix[0] = {{1.0, 2.0, 0.0, 0.0}};
    matrix[1] = {{2.0, 1.0, 2.0, 0.0}};
    matrix[2] = {{0.0, 2.0, 1.0, 2.0}};
    matrix[3] = {{0.0, 0.0, 2.0, 1.0}};

    // create excitation
    std::array<float, 4> b = {{1.0, 0.0, 0.0, 0.0}};

    b = fastEIT::math::gaussElemination<float, 4>(matrix, b);
    EXPECT_FLOAT_EQ(b[0], -1.4);
    EXPECT_FLOAT_EQ(b[1], 1.2);
    EXPECT_FLOAT_EQ(b[2], 0.8);
    EXPECT_FLOAT_EQ(b[3], -1.6);
};
