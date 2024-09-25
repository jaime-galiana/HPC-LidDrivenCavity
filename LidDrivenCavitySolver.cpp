/**
 * @file LidDrivenCavitySolver.cpp
 *
 * High-Performance Computing 2023-24
 *
 * Jaime Galiana Herrera
 * CID - 0177600
 * 
 * Main file
 * 
 */

#include <iostream>
#include <mpi.h>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"

int main(int argc, char **argv)
{
    // Define command-line options
    po::options_description opts("Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(201),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(201),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.005),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(0.1),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(1000),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

   // Parse the command-line arguments using Boost Program Options and store them in a map of options and values
    po::variables_map vm;

    // Parse the command-line arguments and populate the variables_map vm with the parsed values
    po::store(po::parse_command_line(argc, argv, opts), vm);

    // Notify the variables_map vm that parsing is complete and finalize the parsing process
    po::notify(vm);

    // Check if "--help" option is specified, and print help message
    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    // Read variables from vm and save as variables
    const double Lx_global  = vm["Lx"].as<double>();
    const double Ly_global  = vm["Ly"].as<double>();
    const int Nx_global     = vm["Nx"].as<int>();
    const int Ny_global     = vm["Ny"].as<int>();


    // Initialize variables for MPI
    int world_rank = 0;
    int world_size = 0;

    // Initialize MPI process
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
    }

    // Get the rank and size on each process.
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int p = sqrt(world_size);

    // Check if number of precesses are a perfect square
    if ((p * p) != world_size || world_size > 16) {
        if (world_rank == 0) {
            cerr << "Error: Number of processes must be a perfect square and less than 16." << endl;
        }
        MPI_Finalize();
        return 1;
    }    

    // Creates new instance of the LidDrivenCavity class
    LidDrivenCavity* solver = new LidDrivenCavity();
    
    // Create Cartesian grid
    solver->CreateCartesianGrid();

    // Find neighbouring grid points for each grid point
    solver->FindGridNeighbours();

    // Determine local size of the grid for each MPI process
    solver->GridLocalSize(p, Nx_global, Ny_global, Lx_global, Ly_global);

    solver->SetDomainSize(solver->Lx_local, solver->Ly_local);
    solver->SetGridSize(solver->Nx_local, solver->Ny_local);
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());

    solver->Initialise();

    if (solver->grid_rank == 0) {
        solver->PrintConfiguration(Nx_global, Ny_global, Lx_global, Ly_global);
    }

    if (world_size == 1){
        solver->WriteSolution("ic.txt");
    }

    // Solve time step vorticity
    solver->Integrate();

    if (world_size == 1){
        solver->WriteSolution("final.txt");
    }

    // Finilize MPI process
    MPI_Finalize();
    
    delete solver;
	return 0;
} 