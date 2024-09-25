#include <iostream>
#include <algorithm>
#include <cstring>
#include <math.h>
#include <iomanip>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#include "SolverCG.h"
#include "LidDrivenCavity.h"

#define IDX(I,J) ((J)*Nx + (I))


SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    int n = Nx*Ny;
    r = new double[n];
    p = new double[n];
    z = new double[n];
    t = new double[n]; //temp
    r_reduced = new double[(Nx-2)*(Ny-2)];
    b_reduced = new double[(Nx-2)*(Ny-2)];
    t_reduced = new double[(Nx-2)*(Ny-2)];
    p_reduced = new double[(Nx-2)*(Ny-2)];
    z_reduced = new double[(Nx-2)*(Ny-2)];
}

SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
    delete[] r_reduced;
    delete[] p_reduced;
    delete[] z_reduced;
    delete[] b_reduced;
}


void SolverCG::Solve(double* b, double* x, LidDrivenCavity& solver) {
    unsigned int n = Nx*Ny;
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.001;

    reduce_Matrix(b, b_reduced);
    eps = cblas_ddot((Nx-2)*(Ny-2), b_reduced, 1, b_reduced, 1);
    MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_SUM, solver.mygrid);
    eps = sqrt(eps);

    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        if (solver.grid_rank == 0){
            cout << "Norm is " << eps << endl;
        }
        return;
    }

    ApplyOperator(x, t);

    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    ImposeBC(r);

    cblas_daxpy(n, -1.0, t, 1, r, 1);
    Precondition(r, z);
    cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0


    k = 0;

    MPI_Barrier(solver.mygrid);

    do {
        k++;
        // Perform action of Nabla^2 * p
        solver.sendRecvNeighbours(p);

        ApplyOperator(p, t);

        solver.sendRecvNeighbours(t);
        
        reduce_Matrix(t, t_reduced);
        reduce_Matrix(p, p_reduced);

        alpha = cblas_ddot((Nx-2)*(Ny-2), t_reduced, 1, p_reduced, 1);  // alpha = p_k^T A p_k
        MPI_Allreduce(&alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, solver.mygrid);

        reduce_Matrix(r, r_reduced);
        reduce_Matrix(z, z_reduced);
        alpha = cblas_ddot((Nx-2)*(Ny-2), r_reduced, 1, z_reduced, 1)/alpha;
        MPI_Allreduce(&alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, solver.mygrid);

        beta  = cblas_ddot((Nx-2)*(Ny-2), r_reduced, 1, z_reduced, 1);  // z_k^T r_k
        MPI_Allreduce(&beta, &beta, 1, MPI_DOUBLE, MPI_SUM, solver.mygrid);

        cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1);  // r_{k+1} = r_k - alpha_k A p_k
        
        // Reduce matrix to calculate error
        reduce_Matrix(r, r_reduced);
        eps = cblas_ddot((Nx-2)*(Ny-2), r_reduced, 1, r_reduced, 1);
        MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_SUM, solver.mygrid);
        eps = sqrt(eps);

        if (eps < tol*tol) {
            break;
        }

        Precondition(r, z);

        reduce_Matrix(r, r_reduced);
        reduce_Matrix(z, z_reduced);
        beta = cblas_ddot((Nx-2)*(Ny-2), r_reduced, 1, z_reduced, 1)/beta;  // z_k^T r_k
        MPI_Allreduce(&beta, &beta, 1, MPI_DOUBLE, MPI_SUM, solver.mygrid);

        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);

        MPI_Barrier(solver.mygrid);

    } while (k < 5000); // Set a maximum number of iterations 

    if (k == 5000) {
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }

    if (solver.grid_rank == 0){
        cout << "Converged in " << k << " iterations. eps = " << eps << endl;
    }
}

void SolverCG::ApplyOperator(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    #pragma omp parallel for collapse(2) default(none) shared(Nx, Ny, in, out, dx2i, dy2i) schedule(static)
    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i+1, j)])*dx2i
                          + ( -     in[IDX(i, j-1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, j+1)])*dy2i;
        }
    }
}

void SolverCG::Precondition(double* in, double* out) {
    int i, j;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 1/2.0*(dx2i + dy2i);
    #pragma omp parallel for collapse(2) default(none) shared(Nx, Ny, in, out, factor) schedule(static) 
    for (j = 1; j < Ny-1; ++j) {
        for (i = 1; i < Nx-1; ++i) {
            out[IDX(i,j)] = in[IDX(i,j)]*factor;
        }
    }
}

void SolverCG::ImposeBC(double* inout) {
    // Boundaries
    #pragma omp parallel for default(none) shared(Nx, inout, Ny) schedule(static)
    for (int i = 0; i < Nx; ++i) {
        inout[IDX(i, 0)] = 0.0;
        inout[IDX(i, Ny-1)] = 0.0;
    }

    #pragma omp parallel for default(none) shared(Nx, inout, Ny) schedule(static)
    for (int j = 0; j < Ny; ++j) {
        inout[IDX(0, j)] = 0.0;
        inout[IDX(Nx - 1, j)] = 0.0;
    }
}

void SolverCG::reduce_Matrix(double* matrixIn, double* matrixOut){
    #pragma omp parallel for collapse(2) default(none) shared(Nx, Ny, matrixIn, matrixOut) schedule(static) 
    for(int j = 1; j < Ny-1; j++){ // row
        for(int i = 1; i < Nx-1; i++){ // column
            matrixOut[(j-1)*(Nx-2) + (i-1)] = matrixIn[IDX(i,j)];
        }
    }
}