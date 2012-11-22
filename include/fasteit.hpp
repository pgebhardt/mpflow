// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_FASTEIT_HPP
#define FASTEIT_FASTEIT_HPP

// std lib includes
#include <stdexcept>
#include <assert.h>

// cuda includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

// namespace fastEIT
namespace fastEIT {

#include "dtype.hpp"
#include "mesh.hpp"
#include "electrodes.hpp"
#include "linearBasis.hpp"
#include "model.hpp"
#include "conjugate.hpp"
#include "sparseConjugate.hpp"
#include "forward.hpp"
#include "inverse.hpp"
#include "solver.hpp"

};

#endif
