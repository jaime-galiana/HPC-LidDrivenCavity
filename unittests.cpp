/**
 *@file unittests.cpp 
 *
 *@brief File containing unit tests  
 */
#define BOOST_TEST_MODULE unittests
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#include "LidDrivenCavity.h"
#include "SolverCG.h"
using namespace std;


struct MPIFixture {
    public:
        explicit MPIFixture() {
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout << "Initialising MPI" << endl;
            MPI_Init(&argc, &argv);
        }

        ~MPIFixture() {
            cout << "Finalising MPI" << endl;
            MPI_Finalize();
        }

        int argc;
        char **argv;
};

BOOST_GLOBAL_FIXTURE(MPIFixture);

BOOST_AUTO_TEST_CASE(SetReynoldNumberTest){
    LidDrivenCavity solver;
    solver.SetReynoldsNumber(1000.0);
    BOOST_TEST(solver.getReynoldsNumber(), 1000.0);
    BOOST_TEST(solver.getNu(), 0.001);
}

BOOST_AUTO_TEST_CASE(SetDomainSizeTest)
{
    LidDrivenCavity solver;
    solver.SetDomainSize(10.0, 20.0);

    BOOST_CHECK_EQUAL(solver.getDomainSizeX(), 10.0);
    BOOST_CHECK_EQUAL(solver.getDomainSizeY(), 20.0);
}

BOOST_AUTO_TEST_CASE(ReduceMatrixTest)
{
    // Create a SolverCG object
    int Nx = 4; // Example values for Nx and Ny
    int Ny = 4;
    double dx = 1.0; // Example values for dx and dy
    double dy = 1.0;
    SolverCG solver(Nx, Ny, dx, dy);

    // Create input and output matrices
    double* matrixIn = new double[Nx * Ny];
    double* matrixOut = new double[(Nx - 2) * (Ny - 2)]; // Expected output size

    // Fill input matrix with some data (e.g., consecutive integers)
    int num = 0;
    for (int i = 0; i < Nx * Ny; ++i) {
        matrixIn[i] = num++;
    }

    // Call the reduce_Matrix function
    solver.reduce_Matrix(matrixIn, matrixOut);

    // Verify the correctness of the reduced matrix
    int expected_value = 5; // Start from 1, skipping the boundary
    for (int j = 0; j < Ny - 2; ++j) {
        for (int i = 0; i < Nx - 2; ++i) {
            int index = j * (Nx - 2) + i;
            BOOST_TEST(matrixOut[index] == expected_value);
            expected_value++;
        }
        expected_value=expected_value+2;
    }

    // Clean up allocated memory
    delete[] matrixIn;
    delete[] matrixOut;
}