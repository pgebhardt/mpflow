// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_solver_config_t config, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (config == NULL)) {
        return LINALGCU_ERROR;
    }

    // check config
    if ((config->drivePattern == NULL) || (config->measurmentPattern == NULL) ||
        (config->mesh == NULL) || (config->electrodes == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastect_solver_t solver = malloc(sizeof(fastect_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->forwardSolver = NULL;
    solver->inverseSolver = NULL;
    solver->dSigma = NULL;
    solver->sigmaRef = NULL;
    solver->jacobian = NULL;
    solver->calculatedVoltage = NULL;
    solver->measuredVoltage = NULL;
    solver->cublasHandle = NULL;
    solver->linearMode = config->linearMode;

    // create cublas handle
    if (cublasCreate(&solver->cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->dSigma, config->mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&solver->sigmaRef, config->mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&solver->jacobian,
        config->measurmentPattern->columns * config->drivePattern->columns,
        config->mesh->elementCount, stream);
    error |= linalgcu_matrix_create(&solver->calculatedVoltage,
        config->measurmentPattern->columns, config->drivePattern->columns, stream);
    error |= linalgcu_matrix_create(&solver->measuredVoltage,
        config->measurmentCount, config->driveCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // create solver
    error  = fastect_forward_solver_create(&solver->forwardSolver, config->mesh,
        config->electrodes, config->driveCount, config->measurmentCount,
        config->drivePattern, config->measurmentPattern, solver->cublasHandle, stream);
    error |= fastect_inverse_solver_create(&solver->inverseSolver, solver->jacobian,
        config->regularizationFactor, solver->cublasHandle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // init sigma to sigma_0
    for (linalgcu_size_t i = 0; i < config->mesh->elementCount; i++) {
        linalgcu_matrix_set_element(solver->sigmaRef, config->sigma0, i, 0);
    }

    // copy sigma to device
    linalgcu_matrix_copy_to_device(solver->sigmaRef, LINALGCU_TRUE, stream);

    // update forward system matrices
    fastect_grid_update_system_matrix(solver->forwardSolver->grid, solver->sigmaRef, stream);

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t fastect_solver_release(fastect_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_solver_t solver = *solverPointer;

    // cleanup
    fastect_forward_solver_release(&solver->forwardSolver);
    fastect_inverse_solver_release(&solver->inverseSolver);
    linalgcu_matrix_release(&solver->dSigma);
    linalgcu_matrix_release(&solver->sigmaRef);
    linalgcu_matrix_release(&solver->jacobian);
    linalgcu_matrix_release(&solver->calculatedVoltage);
    linalgcu_matrix_release(&solver->measuredVoltage);
    cublasDestroy(solver->cublasHandle);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// pre solve for accurate initial jacobian
linalgcu_error_t fastect_solver_pre_solve(fastect_solver_t solver, cudaStream_t stream) {
    // check input
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // forward solving a few steps
    error |= fastect_forward_solver_solve(solver->forwardSolver, solver->sigmaRef,
        solver->jacobian, solver->calculatedVoltage, 1000, solver->cublasHandle, stream);

    // calc system matrix
    error |= fastect_inverse_solver_calc_system_matrix(solver->inverseSolver, solver->jacobian,
        solver->cublasHandle, stream);

    // set measuredVoltage to calculatedVoltage
    error |= linalgcu_matrix_copy(solver->measuredVoltage, solver->calculatedVoltage,
        LINALGCU_TRUE, NULL);

    return error;
}

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, cudaStream_t stream) {
    // check input
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    if (solver->linearMode == LINALGCU_TRUE) {
        // inverse
        error |= fastect_inverse_solver_solve_linear(solver->inverseSolver,
            solver->jacobian, solver->calculatedVoltage, solver->measuredVoltage,
            solver->dSigma, 90, solver->cublasHandle, stream);
    }
    else {
        // forward
        error  = fastect_forward_solver_solve(solver->forwardSolver, solver->sigmaRef,
            solver->jacobian, solver->calculatedVoltage, 10, solver->cublasHandle, stream);

        // inverse
        error |= fastect_inverse_solver_solve_non_linear(solver->inverseSolver,
            solver->jacobian, solver->calculatedVoltage, solver->measuredVoltage,
            solver->dSigma, 75, solver->cublasHandle, stream);

        // add to sigma
        error |= linalgcu_matrix_add(solver->sigmaRef, solver->dSigma, solver->cublasHandle, stream);
    }

    return error;
}
