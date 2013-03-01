#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

TEST(MathematicsTest, Polar) {
    auto polar = fastEIT::math::polar(std::make_tuple(1.0 / std::sqrt(2.0), 1.0 / std::sqrt(2.0)));
    EXPECT_FLOAT_EQ(std::get<0>(polar), 1.0);
    EXPECT_FLOAT_EQ(std::get<1>(polar), M_PI / 4.0);
};

TEST(MathematicsTest, Kartesian) {
    auto kartesian = fastEIT::math::kartesian(std::make_tuple(2.0, 3.0 * M_PI / 4.0));
    EXPECT_FLOAT_EQ(std::get<0>(kartesian), -std::sqrt(2.0));
    EXPECT_FLOAT_EQ(std::get<1>(kartesian), std::sqrt(2.0));
};

TEST(MathematicsTest, RoundTo) {
    EXPECT_FLOAT_EQ(fastEIT::math::roundTo(0, 10), 0);
    EXPECT_FLOAT_EQ(fastEIT::math::roundTo(1, 4), 4);
    EXPECT_FLOAT_EQ(fastEIT::math::roundTo(7, 3), 9);
};

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
