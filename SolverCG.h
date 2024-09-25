/**
 * @file LidDrivenCavity.h
 *
 * High-Performance Computing 2023-24
 *
 * Header file of LidDrivenCavity.cpp
 *
 * Define all public classes and private variables
 *
 */
#pragma once

#include "LidDrivenCavity.h"

/**
 * @brief The SolverCG class solves a linear system using the Conjugate Gradient method.
 */
class SolverCG
{
public:

    /**
    * @brief Creates solverCG class. Allocates dynamic memory.
    * 
    * @param pNx Number of grid points in the x-direction.
    * @param pNy Number of grid points in the y-direction.
    * @param pdx Spacing in the x-direction.
    * @param pdy Spacing in the y-direction.
    */
    SolverCG(int pNx, int pNy, double pdx, double pdy);

    /**
    * @brief Destructs solverCG class. Deletes dynamic memory on the SolverCG class.
    */
    ~SolverCG();

    /**
    * @brief Solves a linear system using the Conjugate Gradient method.
    * 
    * @param b Right-hand side of the linear system.
    * @param x Initial guess for the solution, updated with the result.
    * @param solver Reference to the LidDrivenCavity object.
    */
    void Solve(double* b, double* x, LidDrivenCavity& solver);    

    /**
    * @brief Function to extract a column from a matrix.
    * 
    * This function extracts a column from a given matrix and stores it in an array.
    * 
    * @param matrix Pointer to the matrix.
    * @param columnIdx Index of the column to be extracted.
    * @param column Pointer to the array where the column will be stored.
    */
    void get_column(double* matrix, int columnIdx, double* column);

    /**
    * @brief Function to extract a column from a matrix.
    * 
    * This function extracts a column from a given matrix and stores it in an array.
    * 
    * @param matrix Pointer to the matrix.
    * @param columnIdx Index of the column to be extracted.
    * @param column Pointer to the array where the column will be stored.
    */
    void get_row(double* matrix, int rowIdx, double* row);

    /**
    * @brief Function to update a column of a matrix.
    * 
    * This function updates a column of a given matrix with the values from an array.
    * 
    * @param matrix Pointer to the matrix.
    * @param columnIdx Index of the column to be updated.
    * @param column Pointer to the array containing the new column values.
    */
    void update_column(double* matrix, int columnIdx, double* column);

    /**
    * @brief Function to update a row of a matrix.
    * 
    * This function updates a row of a given matrix with the values from an array.
    * 
    * @param matrix Pointer to the matrix.
    * @param rowIdx Index of the row to be updated.
    * @param row Pointer to the array containing the new row values.
    */
    void update_row(double* matrix, int rowIdx, double* row);

    /**
    * @brief Reduces a matrix by removing boundary elements.
    * 
    * This function reduces a matrix by removing boundary elements, such as the first and last rows and columns,
    * and stores the result in another matrix.
    * 
    * @param matrixIn Pointer to the input matrix.
    * @param matrixOut Pointer to the output matrix where the reduced matrix will be stored.
    */
    void reduce_Matrix(double* matrixIn, double* matrixOut);

private:
    double dx;   /**< Spacing in the x-direction. */
    double dy;   /**< Spacing in the y-direction. */
    int Nx;      /**< Number of grid points in the x-direction. */
    int Ny;      /**< Number of grid points in the y-direction. */

    double* r;   /**< Residual vector. */
    double* p;   /**< Search direction vector. */
    double* z;   /**< Preconditioned search direction vector. */
    double* t;   /**< Temporal vector. */

    double* b_reduced;   /**< Reduced right-hand side vector. */
    double* r_reduced;   /**< Reduced residual vector. */
    double* t_reduced;   /**< Reduced temporal vector. */
    double* p_reduced;   /**< Reduced search direction vector. */
    double* z_reduced;   /**< Reduced preconditioned search direction vector. */

    /**
    * @brief Applies the discretized Laplacian operator to the input vector.
    * 
    * @param in Input vector.
    * @param out Output vector.
    */
    void ApplyOperator(double* p, double* t);

    /**
    * @brief Applies preconditioning to the input vector.
    * 
    * @param in Input vector.
    * @param out Output vector.
    */
    void Precondition(double* p, double* t);

    /**
    * @brief Imposes boundary condition.
    * 
    * Boudary values equal zero.
    * 
    * @param inout Inout vector to which BC are applied.
    */
    void ImposeBC(double* p);
};