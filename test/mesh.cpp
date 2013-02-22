#include "gtest/gtest.h"
#include "fasteit/fasteit.h"

TEST(MeshTest, Constructor) {
    // create matrices
    auto nodes = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 2, nullptr);
    auto elements = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(10, 2, nullptr);
    auto boundary = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(10, 2, nullptr);

    // create correct mesh
    std::shared_ptr<fastEIT::Mesh> mesh = nullptr;
    EXPECT_NO_THROW({
        mesh = std::make_shared<fastEIT::Mesh>(nodes, elements, boundary, 0.1f, 0.3f);
    });

    // check member
    EXPECT_EQ(mesh->nodes(), nodes);
    EXPECT_EQ(mesh->elements(), elements);
    EXPECT_EQ(mesh->boundary(), boundary);
    EXPECT_EQ(mesh->radius(), 0.1f);
    EXPECT_EQ(mesh->height(), 0.3f);

    // check Constructor errors
    EXPECT_THROW(
        std::make_shared<fastEIT::Mesh>(
            std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(10, 1, nullptr),
        elements, boundary, 0.1f, 0.3f),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::Mesh>(nullptr, elements, boundary, 0.1f, 0.3f),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::Mesh>(nodes, nullptr, boundary, 0.1f, 0.3f),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::Mesh>(nodes, elements, nullptr, 0.1f, 0.3f),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::Mesh>(nodes, elements, boundary, -0.1f, 0.3f),
        std::invalid_argument);
    EXPECT_THROW(
        std::make_shared<fastEIT::Mesh>(nodes, elements, boundary, 0.1f, 0.0f),
        std::invalid_argument);
};

TEST(MeshTest, ElementNodes) {
    // create some nodes
    auto nodes = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 2, nullptr);
    for (fastEIT::dtype::index row = 0; row < nodes->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < nodes->columns(); ++column) {
        (*nodes)(row, column) = (fastEIT::dtype::real)(row + column * nodes->rows());
    }

    // create some elements
    auto elements = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(2, 3, nullptr);
    std::tie((*elements)(0, 0), (*elements)(0, 1), (*elements)(0, 2)) =
        std::make_tuple(0, 1, 2);
    std::tie((*elements)(1, 0), (*elements)(1, 1), (*elements)(1, 2)) =
        std::make_tuple(0, 3, 2);

    // create mesh
    auto mesh = std::make_shared<fastEIT::Mesh>(nodes, elements,
        std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(2, 2, nullptr),
        0.1f, 0.3f);

    // check element nodes
    for (fastEIT::dtype::index element = 0; element < elements->rows(); ++element) {
        // get element nodes
        std::vector<std::tuple<fastEIT::dtype::index, std::tuple<
            fastEIT::dtype::real, fastEIT::dtype::real>>> elementNodes;
        EXPECT_NO_THROW({
            elementNodes = mesh->elementNodes(element);
        });

        // check indices
        EXPECT_EQ(std::get<0>(elementNodes[0]), (*elements)(element, 0));
        EXPECT_EQ(std::get<0>(elementNodes[1]), (*elements)(element, 1));
        EXPECT_EQ(std::get<0>(elementNodes[2]), (*elements)(element, 2));

        // check coordinates
        EXPECT_EQ(std::get<1>(elementNodes[0]), std::make_tuple(
            (*nodes)(std::get<0>(elementNodes[0]), 0),
            (*nodes)(std::get<0>(elementNodes[0]), 1)));
        EXPECT_EQ(std::get<1>(elementNodes[1]), std::make_tuple(
            (*nodes)(std::get<0>(elementNodes[1]), 0),
            (*nodes)(std::get<0>(elementNodes[1]), 1)));
        EXPECT_EQ(std::get<1>(elementNodes[2]), std::make_tuple(
            (*nodes)(std::get<0>(elementNodes[2]), 0),
            (*nodes)(std::get<0>(elementNodes[2]), 1)));
    }

    // check error
    EXPECT_THROW(mesh->elementNodes(elements->rows()), std::invalid_argument);
};

TEST(MeshTest, BoundaryNodes) {
    // create some nodes
    auto nodes = std::make_shared<fastEIT::Matrix<fastEIT::dtype::real>>(4, 2, nullptr);
    for (fastEIT::dtype::index row = 0; row < nodes->rows(); ++row)
    for (fastEIT::dtype::index column = 0; column < nodes->columns(); ++column) {
        (*nodes)(row, column) = (fastEIT::dtype::real)(row + column * nodes->rows());
    }

    // create boundary
    auto boundary = std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(2, 2, nullptr);
    std::tie((*boundary)(0, 0), (*boundary)(0, 1)) =
        std::make_tuple(0, 1);
    std::tie((*boundary)(1, 0), (*boundary)(1, 1)) =
        std::make_tuple(2, 3);

    // create mesh
    auto mesh = std::make_shared<fastEIT::Mesh>(nodes,
        std::make_shared<fastEIT::Matrix<fastEIT::dtype::index>>(2, 2, nullptr),
        boundary, 0.1f, 0.3f);

    // check boundary nodes
    for (fastEIT::dtype::index boundNode = 0; boundNode < boundary->rows(); ++boundNode) {
        // get boundary nodes
        std::vector<std::tuple<fastEIT::dtype::index, std::tuple<
            fastEIT::dtype::real, fastEIT::dtype::real>>> boundaryNodes;
        EXPECT_NO_THROW({
            boundaryNodes = mesh->boundaryNodes(boundNode);
        });

        // check indices
        EXPECT_EQ(std::get<0>(boundaryNodes[0]), (*boundary)(boundNode, 0));
        EXPECT_EQ(std::get<0>(boundaryNodes[1]), (*boundary)(boundNode, 1));

        // check coordinates
        EXPECT_EQ(std::get<1>(boundaryNodes[0]), std::make_tuple(
            (*nodes)(std::get<0>(boundaryNodes[0]), 0),
            (*nodes)(std::get<0>(boundaryNodes[0]), 1)));
        EXPECT_EQ(std::get<1>(boundaryNodes[1]), std::make_tuple(
            (*nodes)(std::get<0>(boundaryNodes[1]), 0),
            (*nodes)(std::get<0>(boundaryNodes[1]), 1)));
    }

    // check error
    EXPECT_THROW(mesh->boundaryNodes(boundary->rows()), std::invalid_argument);
};
