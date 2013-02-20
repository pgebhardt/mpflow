#include "gtest/gtest.h"
#include <cmath>
#include "../include/fasteit.h"

TEST(ElectrodesTest, Constructor) {
    // create electrodes
    std::shared_ptr<fastEIT::Electrodes> electrodes = nullptr;
    EXPECT_NO_THROW({
        electrodes = std::make_shared<fastEIT::Electrodes>(
            20U, std::make_tuple(0.08f, 0.5f), 0.4f);
    });

    // check member
    EXPECT_EQ(electrodes->count(), 20U);
    EXPECT_EQ(electrodes->shape(), std::make_tuple(0.08f, 0.5f));
    EXPECT_EQ(electrodes->impedance(), 0.4f);

    // check invalid arguments
    EXPECT_THROW({
        electrodes = std::make_shared<fastEIT::Electrodes>(
            0U, std::make_tuple(0.08f, 0.5f), 0.4f);
    }, std::invalid_argument);
    EXPECT_THROW({
        electrodes = std::make_shared<fastEIT::Electrodes>(
            20U, std::make_tuple(0.0f, 0.5f), 0.4f);
    }, std::invalid_argument);
    EXPECT_THROW({
        electrodes = std::make_shared<fastEIT::Electrodes>(
            20U, std::make_tuple(0.08f, 0.0f), 0.4f);
    }, std::invalid_argument);
    EXPECT_THROW({
        electrodes = std::make_shared<fastEIT::Electrodes>(
            20U, std::make_tuple(0.08f, 0.5f), 0.0f);
    }, std::invalid_argument);
};

TEST(ElectrodesTest, CircularBoundary) {
    // create electrodes
    auto electrodes = fastEIT::electrodes::circularBoundary(
        32U, std::make_tuple(0.05f, 0.1f), 1.0f, 1.0f);

    // check coordinates
    for (fastEIT::dtype::index electrode = 0; electrode < electrodes->count(); ++electrode) {
        // calc polar coordinates
        auto start = fastEIT::math::polar(std::get<0>(electrodes->coordinates(electrode)));
        auto end = fastEIT::math::polar(std::get<1>(electrodes->coordinates(electrode)));

        // check radius
        EXPECT_LT(std::abs(std::get<0>(start) - std::get<0>(end)), 1e-6);

        // check length
        EXPECT_LT(std::abs(1.0f * std::fmodf(
            std::abs(std::get<1>(start) - std::get<1>(end)), 2.0f * M_PI) - 0.05f), 1e6);
    }

    // check invalid arguments
    EXPECT_THROW({
        electrodes = fastEIT::electrodes::circularBoundary(
            32U, std::make_tuple(0.05f, 0.1f), 1.0f, 0.0f);
    }, std::invalid_argument);
};
