// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

// create basis class
template <
    int template_nodes_per_edge,
    int template_nodes_per_element
>
fastEIT::basis::Basis<template_nodes_per_edge, template_nodes_per_element>::Basis(
    std::array<std::tuple<dtype::real, dtype::real>, template_nodes_per_element> nodes,
    dtype::index one) {
    // init member
    this->nodes_ = nodes;
    for (dtype::real& coefficient : this->coefficients()) {
        coefficient = 0.0f;
    }

    // calc coefficients
    std::array<std::array<dtype::real, template_nodes_per_element>, template_nodes_per_element> A;
    std::array<dtype::real, template_nodes_per_element> b;
}

// specialisations
template class fastEIT::basis::Basis<2, 3>;
template class fastEIT::basis::Basis<3, 6>;
