// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver
linalgcuError_t fastect_solver_create(fastectSolver_t* solverPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuSize_t numHarmonics,
    linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrix_t measurmentPattern, linalgcuMatrix_t drivePattern,
    linalgcuMatrixData_t sigmaRef, linalgcuMatrixData_t regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drivePattern == NULL) || (measurmentPattern == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastectSolver_t solver = malloc(sizeof(fastectSolver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->forwardSolver = NULL;
    solver->calibrationSolver = NULL;
    solver->inverseSolver = NULL;
    solver->dGamma = NULL;
    solver->gamma = NULL;
    solver->jacobian = NULL;
    solver->calculatedVoltage = NULL;
    solver->measuredVoltage = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&solver->dGamma, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&solver->gamma, mesh->elementCount, 1, stream);
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

    // create solver
    error  = fastect_forward_solver_create(&solver->forwardSolver, mesh, electrodes,
        sigmaRef, numHarmonics, driveCount, measurmentCount, drivePattern,
        measurmentPattern, handle, stream);
    error |= fastect_calibration_solver_create(&solver->calibrationSolver, solver->jacobian,
        regularizationFactor, handle, stream);
    error |= fastect_inverse_solver_create(&solver->inverseSolver,
        solver->calibrationSolver->jacobianSquare, solver->jacobian, handle, stream);

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
linalgcuError_t fastect_solver_release(fastectSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectSolver_t solver = *solverPointer;

    // cleanup
    fastect_forward_solver_release(&solver->forwardSolver);
    fastect_calibration_solver_release(&solver->calibrationSolver);
    fastect_inverse_solver_release(&solver->inverseSolver);
    linalgcu_matrix_release(&solver->dGamma);
    linalgcu_matrix_release(&solver->gamma);
    linalgcu_matrix_release(&solver->jacobian);
    linalgcu_matrix_release(&solver->calculatedVoltage);
    linalgcu_matrix_release(&solver->measuredVoltage);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// pre solve for accurate initial jacobian
linalgcuError_t fastect_solver_pre_solve(fastectSolver_t solver, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // forward solving a few steps
    error |= fastect_forward_solver_solve(solver->forwardSolver, solver->jacobian,
        solver->gamma, solver->calculatedVoltage, 1000, handle, stream);

    // calc system matrix
    error |= fastect_calibration_solver_calc_system_matrix(solver->calibrationSolver,
        solver->jacobian, handle, stream);

    // set measuredVoltage to calculatedVoltage
    error |= linalgcu_matrix_copy(solver->measuredVoltage, solver->calculatedVoltage,
        LINALGCU_TRUE, NULL);

    return error;
}

// calibrate
linalgcuError_t fastect_solver_calibrate(fastectSolver_t solver, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // forward
    error  = fastect_forward_solver_solve(solver->forwardSolver, solver->jacobian,
        solver->gamma, solver->calculatedVoltage, 10, handle, stream);

    // calibration
    error |= fastect_calibration_solver_calibrate(solver->calibrationSolver,
        solver->jacobian, solver->calculatedVoltage, solver->measuredVoltage,
        solver->gamma, 75, handle, stream);

    return error;
}

// solving
linalgcuError_t fastect_solver_solve(fastectSolver_t solver, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calibration
    error |= fastect_inverse_solver_solve(solver->inverseSolver,
        solver->jacobian, solver->calculatedVoltage, solver->measuredVoltage,
        solver->dGamma, 90, handle, stream);

    return error;
}
