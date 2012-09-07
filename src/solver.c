// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t measurmentCount,
    linalgcu_size_t driveCount, linalgcu_matrix_t measurmentPattern,
    linalgcu_matrix_t drivePattern, linalgcu_matrix_data_t sigma0,
    linalgcu_matrix_data_t regularizationFactor, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drivePattern == NULL) || (measurmentPattern == NULL)) {
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
    solver->calibrationSolver = NULL;
    solver->inverseSolver = NULL;
    solver->dSigma = NULL;
    solver->sigmaRef = NULL;
    solver->jacobian = NULL;
    solver->calculatedVoltage = NULL;
    solver->measuredVoltage = NULL;
    solver->cublasHandle = NULL;

    // create cublas handle
    if (cublasCreate(&solver->cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->dSigma, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&solver->sigmaRef, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&solver->jacobian,
        measurmentPattern->columns * drivePattern->columns, mesh->elementCount, stream);
    error |= linalgcu_matrix_create(&solver->calculatedVoltage, measurmentPattern->columns,
        drivePattern->columns, stream);
    error |= linalgcu_matrix_create(&solver->measuredVoltage, measurmentCount, driveCount,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // init sigmaRef to sigma0
    for (linalgcu_size_t i = 0; i < mesh->elementCount; i++) {
        linalgcu_matrix_set_element(solver->sigmaRef, sigma0, i, 0);
    }

    // copy sigma to device
    linalgcu_matrix_copy_to_device(solver->sigmaRef, LINALGCU_TRUE, stream);

    // create solver
    error  = fastect_forward_solver_create(&solver->forwardSolver, mesh, electrodes,
        solver->sigmaRef, driveCount, measurmentCount, drivePattern, measurmentPattern,
        solver->cublasHandle, stream);
    error |= fastect_calibration_solver_create(&solver->calibrationSolver, solver->jacobian,
        regularizationFactor, solver->cublasHandle, stream);
    error |= fastect_inverse_solver_create(&solver->inverseSolver,
        solver->calibrationSolver->systemMatrix, solver->jacobian, solver->cublasHandle,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

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
    fastect_calibration_solver_release(&solver->calibrationSolver);
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
    error |= fastect_calibration_solver_calc_system_matrix(solver->calibrationSolver,
        solver->jacobian, solver->cublasHandle, stream);

    // set measuredVoltage to calculatedVoltage
    error |= linalgcu_matrix_copy(solver->measuredVoltage, solver->calculatedVoltage,
        LINALGCU_TRUE, NULL);

    return error;
}

// calibrate
linalgcu_error_t fastect_solver_calibrate(fastect_solver_t solver, cudaStream_t stream) {
    // check input
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // forward
    error  = fastect_forward_solver_solve(solver->forwardSolver, solver->sigmaRef,
        solver->jacobian, solver->calculatedVoltage, 10, solver->cublasHandle, stream);

    // calibration
    error |= fastect_calibration_solver_calibrate(solver->calibrationSolver,
        solver->jacobian, solver->calculatedVoltage, solver->measuredVoltage,
        solver->sigmaRef, 75, solver->cublasHandle, stream);

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

    // calibration
    error |= fastect_inverse_solver_solve(solver->inverseSolver,
        solver->jacobian, solver->calculatedVoltage, solver->measuredVoltage,
        solver->dSigma, 90, solver->cublasHandle, stream);

    return error;
}
