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

#include <string>
using namespace std;

class SolverCG;

class LidDrivenCavity
{
public:
    MPI_Comm mygrid;
    int grid_rank;
    int coords[2];
    int rank_left;
    int rank_right;
    int rank_top;
    int rank_bottom; 
    int n_Xneighbours   = 0;
    int n_Yneighbours   = 0;
    bool boundaries[4]  = {false, false, false, false}; // left, right, bottom, top
    int Nx_local;
    int Ny_local;
    double Lx_local;
    double Ly_local;


    /**
     * @brief Constructor for the LidDrivenCavity class.
     */
    LidDrivenCavity();

    /**
     * @brief Destructor for the LidDrivenCavity class.
     */
    ~LidDrivenCavity();

    /**
    * @brief Create a Cartesian grid communicator for the Lid-Driven Cavity simulation.
    *
    * Creates a Cartesian grid communicator using MPI for the Lid-Driven Cavity simulation.
    * Checks if MPI has been initialized. If not, it throws an exception.
    *
    * @throws std::exception if MPI is not initialized.
    */
    void CreateCartesianGrid();

    /**
    * @brief Find neighboring processes in the Cartesian grid for the Lid-Driven Cavity simulation.
    *
    * Finds neighboring processes in the Cartesian grid created for the Lid-Driven Cavity simulation.
    * It determines the rank coordinates and identifies neighboring ranks along the x-axis (left/right) and y-axis (up/down).
    *
    * @throws std::runtime_error if it fails to determine neighboring ranks.
    */
    void FindGridNeighbours();
    
    /**
    * @brief Determine local grid size for each process in the Lid-Driven Cavity simulation.
    *
    * Calculates the local grid size for each process in the Lid-Driven Cavity simulation based
    * on the global grid size and the number of processes in each dimension. It considers any additional columns
    * or rows required to evenly distribute the grid across processes.
    *
    * @param p The number of processes in each dimension.
    * @param Nx_global The global grid size in the x-direction.
    * @param Ny_global The global grid size in the y-direction.
    * @param Lx_global The global domain size in the x-direction.
    * @param Ly_global The global domain size in the y-direction.
    */
    void GridLocalSize(const int p, const int Nx_global, const int Ny_global, const double Lx_global, const double Ly_global);

    /**
     * @brief Set the size of the computational domain.
     * 
     * @param xlen Length of the domain in the x-direction.
     * @param ylen Length of the domain in the y-direction.
     */
    void SetDomainSize(double xlen, double ylen);

    /**
     * @brief Set the grid size for the computational domain.
     * 
     * @param nx Number of grid points in x-direction.
     * @param ny Number of grid points in y-direction.
     */
    void SetGridSize(int nx, int ny);

    /**
     * @brief Set the time step size for time integration.
     * 
     * @param deltat Time step size.
     */
    void SetTimeStep(double deltat);

    /**
     * @brief Set the final time for simulation.
     * 
     * @param finalt Final time.
     */
    void SetFinalTime(double finalt);

    /**
     * @brief Set the Reynolds number for simulation.
     * 
     * @param Re Reynolds number.
     */
    void SetReynoldsNumber(double Re);

    /**
     * @brief test get Reynolds number.
     */
    double getReynoldsNumber();

    /**
     * @brief test get nu.
     */
    double getNu();

    /**
     * @brief test get Lx.
     */
    double getDomainSizeX();

    /**
     * @brief test get Ly.
     */
    double getDomainSizeY();

    /**
     * @brief Initialize the simulation.
     * 
     * Allocates memory for arrays and initializes solver objects.
     */
    void Initialise();

    /**
     * @brief Integrate the simulation over time.
     * 
     * Advances the simulation in time using the specified time step size
     * until reaching the final time.
     */
    void Integrate();
    
    /**
     * @brief Write the solution to a file.
     * 
     * @param file Name of the file where solution is written.
     */
    void WriteSolution(std::string file);

    /**
     * @brief Prints to terminal simulation configuration. Prints error if time-step restriction is not satisfied.
     */
    void PrintConfiguration(int Nx_global, int Ny_global, double Lx_global, double Ly_global);

    /**
    * @brief Sends and receives data with neighboring MPI processes based on the presence of walls.
    *
    * @param x Pointer to the data array.
    */
    void sendRecvNeighbours(double* x);

    /**
    * @brief Sends and receives boundary condition data with neighboring MPI processes.
    *
    * This function sends and receives boundary condition data with neighboring MPI processes. 
    * If there is a wall on a particular boundary, boundary conditions are applied directly 
    * to the data. Otherwise, data is exchanged with the adjacent MPI process.
    *
    * @param x Pointer to the data array.
    */
    void sendRecvBC(double* x);

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


    // void PrintMatrix(double* matrix, int Nx, int Ny);

private:
    double* v    = nullptr;
    double* vnew = nullptr;
    double* s    = nullptr;
    double* tmp  = nullptr;

    double dt   = 0.01;
    double T    = 1.0;
    double dx;
    double dy;
    double dxi;
    double dyi;
    double dx2i;
    double dy2i;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;

    SolverCG* cg = nullptr;

    /**
    * @brief Clean up memory allocation.
    */
    void CleanUp();

    /**
    * @brief Update dx and dy based on the current grid size and domain size.
    */
    void UpdateDxDy();

    /**
    * @brief Advance the simulation in time.
    * 
    * 1. Updates the current vorticity at time t on the boundaries.
    * 2. Updated the current vorticity at time t in the interior.
    * 3. Computes new vorticity at time t+dt in the interion.
    * 4. Computes new stream function at time (t+dt) by solving the Poissons equation.
    */
    void Advance();
};

