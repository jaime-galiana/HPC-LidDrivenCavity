/**
 * @file LidDrivenCavity.cpp
 *
 * High-Performance Computing 2023-24
 *
 * Class file of LidDrivenCavity.cpp
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#include "LidDrivenCavity.h"
#include "SolverCG.h"

#define IDX(I,J) ((J)*Nx + (I))

LidDrivenCavity::LidDrivenCavity() 
{
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    // Check if domain is positive
    if (xlen <= 0 || ylen <= 0) {
        cerr << "Error: Domain size must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    // Check if discretization is positive
    if (nx <= 0 || ny <= 0) {
        cerr << "Error: Grid size must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();
}

void LidDrivenCavity::SetTimeStep(double deltat)
{
    // Check if time step size is positive
    if (deltat <= 0) {
        cerr << "Error: Time step must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    this->dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
    // Check if final time is positive
    if (finalt <= 0) {
        cerr << "Error: Final time must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    this->T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{
    // Check if Reynolds number is positive
    if (re <= 0) {
        cerr << "Error: Reynolds number must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    this->Re = re;
    this->nu = 1.0/re;
}

double LidDrivenCavity::getReynoldsNumber(){
    return Re;
}

double LidDrivenCavity::getNu(){
    return nu;
}

double LidDrivenCavity::getDomainSizeX(){
    return Lx;
}

double LidDrivenCavity::getDomainSizeY() {
    return Ly;
}

void LidDrivenCavity::CreateCartesianGrid()
{
    int MPI_init;
    MPI_Initialized(&MPI_init);

    if (!MPI_init) {
        cout << "Error: MPI not initialised" << endl;
        throw exception();
    } else {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        // Create Cartesian grid
        int p              = sqrt(world_size);     // Number of processes per dimension 
        int sizes[2]    = {p, p};               // Size of each grid dimension
        int periods[2]  = {0, 0};               // Non-periodic grid
        int reorder        = 1;                    // Enable reordering

        int err = MPI_Cart_create(MPI_COMM_WORLD, 2, sizes, periods, reorder, &mygrid);
        if (err != MPI_SUCCESS) {
            cout << "Failed to create Cartesian grid" << endl;
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }
}

void LidDrivenCavity::FindGridNeighbours()
{
    // Define grid rank and find rank coordinates
    MPI_Comm_rank(mygrid, &grid_rank);
    MPI_Cart_coords(mygrid, grid_rank, 2, coords);

    // Check neighbours left/right (x-axis)
    int err = MPI_Cart_shift(mygrid, 0, 1, &rank_left, &rank_right);
    if (err == MPI_SUCCESS) {
        if (rank_left == MPI_PROC_NULL) {
            this->boundaries[0] = true;
        } else {
            this->n_Xneighbours++;
        }
        if (rank_right == MPI_PROC_NULL) {
            this->boundaries[1] = true;
        } else {
            this->n_Xneighbours++;
        }
    } else {
        cout << "Failed to determine neighboring ranks along y-axis" << endl;
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Check neighbours up/down (y-axis)
    err = MPI_Cart_shift(mygrid, 1, 1, &rank_bottom, &rank_top);
    if (err == MPI_SUCCESS) {
        if (rank_bottom == MPI_PROC_NULL) {
            this->boundaries[2] = true;
        } else {
            this->n_Yneighbours++;
        }
        if (rank_top == MPI_PROC_NULL) {
            this->boundaries[3] = true;
        } else {
            this->n_Yneighbours++;
        }        
    } else {
        cout << "Failed to determine neighboring ranks along x-axis" << endl;
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
}

void LidDrivenCavity::GridLocalSize(const int p, const int Nx_global, const int Ny_global, const double Lx_global, const double Ly_global)
{
    int rx          = Nx_global % p;             // remainder x
    int ry          = Ny_global % p;             // remainder y

    int addCol_Nx   = ((coords[0] < rx) ? 1 : 0);
    int addRow_Ny   = ((coords[1] < ry) ? 1 : 0);

    this->Nx_local    = (Nx_global / p) + (n_Xneighbours + addCol_Nx);
    this->Ny_local    = (Ny_global / p) + (n_Yneighbours + addRow_Ny);
    this->Lx_local = ((Lx_global) / (Nx_global-1)) * (Nx_local-1);
    this->Ly_local = ((Ly_global) / (Ny_global-1)) * (Ny_local-1); 
}

void LidDrivenCavity::Initialise()
{
    CleanUp();

    v    = new double[Npts]();
    vnew = new double[Npts]();
    s    = new double[Npts]();
    tmp  = new double[Npts]();
    cg   = new SolverCG(Nx, Ny, dx, dy);
}

void LidDrivenCavity::Integrate(){
    int NSteps = ceil(T/dt);
    for (int t = 0; t < NSteps; ++t)
    {
        if (grid_rank == 0) {
            std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*dt
                  << std::endl;
        }               
        Advance();
    }
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    double* u0 = new double[Nx*Ny]();
    double* u1 = new double[Nx*Ny]();

    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy;
            u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0[IDX(i,Ny-1)] = U;
    }

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i)
        {
            k = IDX(i, j);
            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] 
              << " " << u0[k] << " " << u1[k] << std::endl;
        }
        f << std::endl;
    }
    f.close();

    delete[] u0;
    delete[] u1;
}


void LidDrivenCavity::PrintConfiguration(int Nx_global, int Ny_global, double Lx_global, double Ly_global)
{
    cout << "Grid size: " << Nx_global << " x " << Ny_global << endl;
    cout << "Spacing:   " << Ly_global / (Ny_global-1) << " x " << Ly_global / (Ny_global-1) << endl;
    cout << "Length:    " << Lx_global << " x " << Ly_global << endl;
    cout << "Grid pts:  " << Nx_global*Ny_global << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}

void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] vnew;
        delete[] s;
        delete[] tmp;
        delete cg;
    }
}

void LidDrivenCavity::UpdateDxDy()
{
    dx   = Lx / (Nx-1);
    dy   = Ly / (Ny-1);
    dxi  = 1.0/dx;
    dyi  = 1.0/dy;
    dx2i = 1.0/dx/dx;
    dy2i = 1.0/dy/dy;
    Npts = Nx * Ny;
}

void LidDrivenCavity::Advance(){  
    

    // 1. Boundary node vorticity
    // Communicate with neighbour nodes and update stream function s
    sendRecvNeighbours(s);

    // 2. Compute interior vorticity
    #pragma omp parallel for collapse(2)  default(none) shared(Nx, Ny, v, s, dx2i, dy2i) schedule(static)
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            v[IDX(i,j)] = dx2i*(
                    2.0 * s[IDX(i,j)] - s[IDX(i+1,j)] - s[IDX(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[IDX(i,j)] - s[IDX(i,j+1)] - s[IDX(i,j-1)]);
        }
    }

    // Apply BCs in the walls and communicate with the neighbour nodes to update vorticity v
    sendRecvBC(v);

    // 3. Time advance vorticity
    #pragma omp parallel for collapse(2) default(none) shared(Nx, Ny, vnew, v, s, dxi, dyi, dx2i, dy2i, nu) schedule(static)
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            vnew[IDX(i,j)] = v[IDX(i,j)] + dt*(
                ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dxi
                 *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dyi
                 *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dx2i
              + nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dy2i);
        }
    }

    // Communicate with neighbour nodes and update new vorticty vnew
    sendRecvNeighbours(vnew);

    // Sinusoidal test case with analytical solution, which can be used to test
    // the Poisson solver
    /*
    const int k = 3;
    const int l = 3;
    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            vnew[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    */

    // Wait for all processes to finish
    MPI_Barrier(mygrid);

    // Solve Poisson problem using Conjugate Gradient
    cg->Solve(vnew, s, *this);
}

void::LidDrivenCavity::sendRecvNeighbours(double* x)
{
    double* send_col = new double[Ny];
    double* recv_col = new double[Ny];
    double* send_row = new double[Nx];
    double* recv_row = new double[Nx];

    // Check if there is a wall on the left
    if (!boundaries[0]) {
        // There is no wall on the left, share information with the rank on the right
        get_column(x, 1, send_col);
        MPI_Send(send_col, Ny, MPI_DOUBLE, rank_left, 0, mygrid);
        MPI_Recv(recv_col, Ny, MPI_DOUBLE, rank_left, 0, mygrid, MPI_STATUS_IGNORE);
        update_column(x, 0, recv_col);
    }

    // Check if there is a wall on the right
    if (!boundaries[1]) {
        // There is no wall on the right, share information with the rank on the right
        get_column(x, Nx-2, send_col);
        MPI_Send(send_col, Ny, MPI_DOUBLE, rank_right, 0, mygrid);
        MPI_Recv(recv_col, Ny, MPI_DOUBLE, rank_right, 0, mygrid, MPI_STATUS_IGNORE);
        update_column(x, Nx-1, recv_col);
    }

    // Check if there is a wall on the bottom
    if (!boundaries[2]) {
        // There is no wall on the bottom, share information with the rank on the bottom
        get_row(x, 1, send_row);
        MPI_Send(send_row, Nx, MPI_DOUBLE, rank_bottom, 0, mygrid);
        MPI_Recv(recv_row, Nx, MPI_DOUBLE, rank_bottom, 0, mygrid, MPI_STATUS_IGNORE);
        update_row(x, 0, recv_row);
    }

    // Check if there is a wall on the top
    if (!boundaries[3]) {
        // There is no wall on the top, share information with the rank on the top
        get_row(x, Ny-2, send_row);
        MPI_Send(send_row, Nx, MPI_DOUBLE, rank_top, 0, mygrid);
        MPI_Recv(recv_row, Nx, MPI_DOUBLE, rank_top, 0, mygrid, MPI_STATUS_IGNORE);
        update_row(x, Ny-1, recv_row);
    }
    delete[] send_col;
    delete[] recv_col;
    delete[] send_row;
    delete[] recv_row;
}

void LidDrivenCavity::sendRecvBC(double* x)
{
    double* send_col = new double[Ny];
    double* recv_col = new double[Ny];
    double* send_row = new double[Nx];
    double* recv_row = new double[Nx];

    if (boundaries[0]) {
        // There is a wall on the left >> apply BC
        #pragma omp parallel for default(none) shared(Nx, Ny, v, s, dx2i) schedule(static)
        for (int j = 1; j < Ny-1; ++j) {
        // left BC
            v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);
        }
    } else {
        // There is no wall on the left, share information with the rank on the left
        get_column(x, 1, send_col);
        MPI_Send(send_col, Ny, MPI_DOUBLE, rank_left, 0, mygrid);
        MPI_Recv(recv_col, Ny, MPI_DOUBLE, rank_left, 0, mygrid, MPI_STATUS_IGNORE);
        update_column(x, 0, recv_col);
    }

    // Check if there is a wall on the right
    if (boundaries[1]) {
        // There is a wall on the right >> apply BC
        #pragma omp parallel for default(none) shared(Nx, Ny, v, s, dx2i) schedule(static)
        for (int j = 1; j < Ny-1; ++j) {
        // right
            v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - s[IDX(Nx-2,j)]);
        } 
    } else {
        // There is no wall on the right, share information with the rank on the right
        get_column(x, Nx-2, send_col);
        MPI_Send(send_col, Ny, MPI_DOUBLE, rank_right, 0, mygrid);
        MPI_Recv(recv_col, Ny, MPI_DOUBLE, rank_right, 0, mygrid, MPI_STATUS_IGNORE); 
        update_column(x, Nx-1, recv_col);
    }

    // Check if there is a wall on the bottom
    if (boundaries[2]) {
        // There is a wall on the bottom >> apply BC
        #pragma omp parallel for default(none) shared(Nx, Ny, v, s, dy2i) schedule(static)
        for (int i = 1; i < Nx-1; ++i) {
        // top BC
            v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]);
        }
        
    } else {
        // There is no wall on the bottom, share information with the rank on the bottom
        get_row(x, 1, send_row);
        MPI_Send(send_row, Nx, MPI_DOUBLE, rank_bottom, 0, mygrid);
        MPI_Recv(recv_row, Nx, MPI_DOUBLE, rank_bottom, 0, mygrid, MPI_STATUS_IGNORE);
        update_row(x, 0, recv_row);
    }

    // Check if there is a wall on the top
    if (boundaries[3]) {
        // There is a wall on the top >> apply BC
        #pragma omp parallel for default(none) shared(Nx, Ny, v, s, dyi, dy2i) schedule(static)
        for (int i = 1; i < Nx-1; ++i) {
            // bottom BC
            v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - s[IDX(i,Ny-2)])
                       - 2.0 * dyi*U;
        }
    } else {
        // There is no wall on the top, share information with the rank on the top
        get_row(x, Ny-2, send_row);
        MPI_Send(send_row, Nx, MPI_DOUBLE, rank_top, 0, mygrid);
        MPI_Recv(recv_row, Nx, MPI_DOUBLE, rank_top, 0, mygrid, MPI_STATUS_IGNORE);
        update_row(x, Ny-1, recv_row);
    }
    delete[] send_col;
    delete[] recv_col;
    delete[] send_row;
    delete[] recv_row;
}

void LidDrivenCavity::get_column(double* matrix, int columnIdx, double* column) {
    for (int j = 0; j < Ny; j++) {
        column[j] = matrix[IDX(columnIdx, j)];
    }
}

void LidDrivenCavity::get_row(double* matrix, int rowIdx, double* row) {
    for (int i = 0; i < Nx; i++) {
        row[i] = matrix[IDX(i, rowIdx)];
    }
}

void LidDrivenCavity::update_column(double* matrix, int columnIdx, double* column) {
    for (int i = 0; i < Ny; i++) {
        matrix[IDX(columnIdx, i)] = column[i];
    }
}

void LidDrivenCavity::update_row(double* matrix, int rowIdx, double* row) {
    for (int j = 0; j < Nx; j++) {
        matrix[IDX(j, rowIdx)] = row[j];
    }
}

