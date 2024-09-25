CCX      = mpicxx -Wall -fopenmp -O3
TARGET   = solver      # Name of the executable
HDRSCG   = SolverCG.h  # Header files for SolverCG
HDRSLDC  = LidDrivenCavity.h  # Header files for LidDrivenCavity
LDLIBS   = -lblas -lboost_program_options
TESTLIBS = -lboost_unit_test_framework
TESTS    = unittests

# Default target
default: $(TARGET)

# Compile source files into object files
%.o: %.cpp $(HDRSCG) $(HDRSLDC)
	$(CCX) -c $< -o $@

unittests.o: unittests.cpp
	$(CCX) -c $<

# Link object files into the final executable
$(TARGET): LidDrivenCavitySolver.o LidDrivenCavity.o SolverCG.o
	$(CCX) $^ -o $@ $(LDLIBS)

unittests: unittests.o LidDrivenCavity.o SolverCG.o
	$(CCX) -o $(TESTS) $^ $(LDLIBS) $(TESTLIBS)

# Generate documentation using Doxygen
doc:
	doxygen Doxyfile

# Clean up intermediate object files and executables
clean:
	rm -f $(TARGET) $(TESTS) *.o