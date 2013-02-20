#include "gtest/gtest.h"
#include "../include/fasteit.h"

// test class
class ElectrodesTest :
    public ::testing::Test {
protected:
    void SetUp() {
        // create empty mesh
        auto nodes = std::shared_ptr<fastEIT::Matrix::real>>(1, 2);
        auto elements = std::shared_ptr<fastEIT::Matrix::real>>(1, 2);
        auto  = std::shared_ptr<fastEIT::Matrix::real>>(1, 2);
        // create electrodes
        this->electrodes_32_ = std::make_shared<fastEIT::Electrodes>(
            32, std::make_tuple(0.05, 0.1), 
    }

    // member
    std::shared_ptr<fastEIT::Electrodes> electrodes_32_;
    std::shared_ptr<fastEIT::Electrodes> electrodes_16_;
};

