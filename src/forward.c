// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create forward_solver
linalgcuError_t fastect_forward_solver_create(fastectForwardSolver_t* solverPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, linalgcuSize_t driveCount, linalgcuSize_t measurmentCount,
    linalgcuMatrix_t drivePattern, linalgcuMatrix_t measurmentPattern, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drivePattern == NULL) || (measurmentPattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastectForwardSolver_t solver = malloc(sizeof(fastectForwardSolver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->grid = NULL;
    solver->driveSolver = NULL;
    solver->measurmentSolver = NULL;
    solver->drivePhi = NULL;
    solver->measurmentPhi = NULL;
    solver->driveF = NULL;
    solver->measurmentF = NULL;
    solver->voltageCalculation = NULL;

    // create grid
    error = fastect_grid_create(&solver->grid, mesh, electrodes, sigmaRef, numHarmonics,
        handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create conjugate solver
    error  = fastect_conjugate_sparse_solver_create(&solver->driveSolver,
        mesh->vertexCount, driveCount, handle, stream);
    error |= fastect_conjugate_sparse_solver_create(&solver->measurmentSolver,
        mesh->vertexCount, measurmentCount, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create matrix buffer
    solver->drivePhi = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    solver->measurmentPhi = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    solver->driveF = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    solver->measurmentF = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));

    // check success
    if ((solver->drivePhi == NULL) || (solver->measurmentPhi == NULL) ||
        (solver->driveF == NULL) || (solver->measurmentF == NULL)) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // create matrices
    for (linalgcuSize_t i = 0; i < numHarmonics + 1; i++) {
        error |= linalgcu_matrix_create(&solver->drivePhi[i], mesh->vertexCount,
            driveCount, stream);
        error |= linalgcu_matrix_create(&solver->measurmentPhi[i], mesh->vertexCount,
            measurmentCount, stream);
        error |= linalgcu_matrix_create(&solver->driveF[i], mesh->vertexCount,
            driveCount, stream);
        error |= linalgcu_matrix_create(&solver->measurmentF[i], mesh->vertexCount,
            measurmentCount, stream);
    }

    error |= linalgcu_matrix_create(&solver->voltageCalculation,
        measurmentPattern->columns, mesh->vertexCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    for (linalgcuSize_t n = 0; n < numHarmonics + 1; n++) {
        // Run multiply once more to avoid cublas error
        linalgcu_matrix_multiply(solver->driveF[n], solver->grid->excitationMatrix,
            drivePattern, handle, stream);
        error |= linalgcu_matrix_multiply(solver->driveF[n], solver->grid->excitationMatrix,
            drivePattern, handle, stream);

        linalgcu_matrix_multiply(solver->measurmentF[n], solver->grid->excitationMatrix,
            measurmentPattern, handle, stream);
        error |= linalgcu_matrix_multiply(solver->measurmentF[n],
            solver->grid->excitationMatrix, measurmentPattern, handle, stream);
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    error |= linalgcu_matrix_scalar_multiply(solver->driveF[0],
        1.0f / solver->grid->mesh->height, stream);
    error |= linalgcu_matrix_scalar_multiply(solver->measurmentF[0],
        1.0f / solver->grid->mesh->height, stream);

    // calc harmonics
    for (linalgcuSize_t n = 1; n < numHarmonics + 1; n++) {
        error |= linalgcu_matrix_scalar_multiply(solver->driveF[n],
            2.0f * sin(n * M_PI * solver->grid->electrodes->height /
            solver->grid->mesh->height) /
            (n * M_PI * solver->grid->electrodes->height), stream);
        error |= linalgcu_matrix_scalar_multiply(solver->measurmentF[n],
            2.0f * sin(n * M_PI * solver->grid->electrodes->height /
            solver->grid->mesh->height) /
            (n * M_PI * solver->grid->electrodes->height), stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc voltage calculation matrix
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        solver->grid->excitationMatrix->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        solver->grid->excitationMatrix->deviceData, solver->grid->excitationMatrix->rows,
        &beta, solver->voltageCalculation->deviceData, solver->voltageCalculation->rows)
        != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fastect_forward_solver_release(fastectForwardSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectForwardSolver_t solver = *solverPointer;

    // cleanup
    if (solver->drivePhi != NULL) {
        for (linalgcuSize_t i = 0; i < solver->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&solver->drivePhi[i]);
        }
        free(solver->drivePhi);
    }
    if (solver->measurmentPhi != NULL) {
        for (linalgcuSize_t i = 0; i < solver->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&solver->measurmentPhi[i]);
        }
        free(solver->measurmentPhi);
    }
    if (solver->driveF != NULL) {
        for (linalgcuSize_t i = 0; i < solver->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&solver->driveF[i]);
        }
        free(solver->driveF);
    }
    if (solver->measurmentF != NULL) {
        for (linalgcuSize_t i = 0; i < solver->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&solver->measurmentF[i]);
        }
        free(solver->measurmentF);
    }
    fastect_grid_release(&solver->grid);
    fastect_conjugate_sparse_solver_release(&solver->driveSolver);
    fastect_conjugate_sparse_solver_release(&solver->measurmentSolver);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcuError_t fastect_forward_solver_solve(fastectForwardSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t gamma, linalgcuMatrix_t voltage,
    linalgcuSize_t steps, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (gamma == NULL) || (jacobian == NULL) || (voltage == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // update system matrix
    error  = fastect_grid_update_system_matrices(solver->grid, gamma, handle, stream);

    // solve for ground mode
    // solve for drive phi
    error |= fastect_conjugate_sparse_solver_solve_regularized(solver->driveSolver,
        solver->grid->systemMatrices[0], solver->drivePhi[0], solver->driveF[0],
        steps, handle, stream);

    // solve for measurment phi
    error |= fastect_conjugate_sparse_solver_solve_regularized(solver->measurmentSolver,
        solver->grid->systemMatrices[0], solver->measurmentPhi[0], solver->measurmentF[0],
        steps, handle, stream);

    // solve for higher harmonics
    for (linalgcuSize_t n = 1; n < solver->grid->numHarmonics + 1; n++) {
        // solve for drive phi
        error |= fastect_conjugate_sparse_solver_solve(solver->driveSolver,
            solver->grid->systemMatrices[n], solver->drivePhi[n], solver->driveF[n],
            steps, handle, stream);

        // solve for measurment phi
        error |= fastect_conjugate_sparse_solver_solve(solver->measurmentSolver,
            solver->grid->systemMatrices[n], solver->measurmentPhi[n], solver->measurmentF[n],
            steps, handle, stream);
    }

    // calc jacobian
    error |= fastect_forward_solver_calc_jacobian(solver, jacobian, gamma, 0,
        LINALGCU_FALSE, stream);
    for (linalgcuSize_t n = 1; n < solver->grid->numHarmonics + 1; n++) {
        error |= fastect_forward_solver_calc_jacobian(solver, jacobian, gamma, n,
            LINALGCU_TRUE, stream);
    }

    // calc voltage
    error |= linalgcu_matrix_multiply(voltage, solver->voltageCalculation,
        solver->drivePhi[0], handle, stream);

    // add harmonic voltages
    linalgcuMatrixData_t alpha = 1.0f, beta = 1.0f;
    for (linalgcuSize_t n = 1; n < solver->grid->numHarmonics + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, solver->voltageCalculation->rows,
            solver->drivePhi[n]->columns, solver->voltageCalculation->columns, &alpha,
            solver->voltageCalculation->deviceData, solver->voltageCalculation->rows,
            solver->drivePhi[n]->deviceData, solver->drivePhi[n]->rows, &beta,
            voltage->deviceData, voltage->rows);
    }

    return error;
}
