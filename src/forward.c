// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_matrix_t sigma,
    linalgcu_size_t driveCount, linalgcu_size_t measurmentCount, linalgcu_matrix_t drivePattern,
    linalgcu_matrix_t measurmentPattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) || (sigma == NULL) ||
        (drivePattern == NULL) || (measurmentPattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastect_forward_solver_t solver = malloc(sizeof(fastect_forward_solver_s));

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
    error = fastect_grid_create(&solver->grid, mesh, electrodes, sigma, 0, handle, stream);

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

    // create matrices
    error  = linalgcu_matrix_create(&solver->drivePhi, mesh->vertexCount,
        driveCount, stream);
    error |= linalgcu_matrix_create(&solver->measurmentPhi, mesh->vertexCount,
        measurmentCount, stream);
    error |= linalgcu_matrix_create(&solver->driveF, mesh->vertexCount,
        driveCount, stream);
    error |= linalgcu_matrix_create(&solver->measurmentF, mesh->vertexCount,
        measurmentCount, stream);
    error |= linalgcu_matrix_create(&solver->voltageCalculation,
        measurmentPattern->columns, mesh->vertexCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    // Run multiply once more to avoid cublas error
    linalgcu_matrix_multiply(solver->driveF, solver->grid->excitationMatrix, drivePattern,
        handle, stream);
    error  = linalgcu_matrix_multiply(solver->driveF, solver->grid->excitationMatrix,
        drivePattern, handle, stream);
    linalgcu_matrix_multiply(solver->measurmentF, solver->grid->excitationMatrix,
        measurmentPattern, handle, stream);
    error |= linalgcu_matrix_multiply(solver->measurmentF, solver->grid->excitationMatrix,
        measurmentPattern, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc voltage calculation matrix
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
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
linalgcu_error_t fastect_forward_solver_release(fastect_forward_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_forward_solver_t solver = *solverPointer;

    // cleanup
    fastect_grid_release(&solver->grid);
    fastect_conjugate_sparse_solver_release(&solver->driveSolver);
    fastect_conjugate_sparse_solver_release(&solver->measurmentSolver);
    linalgcu_matrix_release(&solver->drivePhi);
    linalgcu_matrix_release(&solver->measurmentPhi);
    linalgcu_matrix_release(&solver->driveF);
    linalgcu_matrix_release(&solver->measurmentF);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    linalgcu_matrix_t sigma, linalgcu_matrix_t jacobian, linalgcu_matrix_t voltage,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (sigma == NULL) || (jacobian == NULL) || (voltage == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // update system matrix
    error  = fastect_grid_update_system_matrices(solver->grid, sigma, handle, stream);

    // solve for drive phi
    error |= fastect_conjugate_sparse_solver_solve_regularized(solver->driveSolver,
        solver->grid->systemMatrices[0], solver->drivePhi, solver->driveF,
        steps, handle, stream);

    // solve for measurment phi
    error |= fastect_conjugate_sparse_solver_solve_regularized(solver->measurmentSolver,
        solver->grid->systemMatrices[0], solver->measurmentPhi, solver->measurmentF,
        steps, handle, stream);

    // calc jacobian
    error |= fastect_forward_solver_calc_jacobian(solver, jacobian, stream);

    // calc voltage
    error |= linalgcu_matrix_multiply(voltage, solver->voltageCalculation,
        solver->drivePhi, handle, stream);

    return error;
}
