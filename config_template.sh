#!/bin/sh

#C++ compiler
CXX_COMPILER="g++"

# C++ flags
CXX_FLAGS="-O3 -flto -Wall -Wno-unused-function"

# add coverage flags
USE_COVERAGE=0

# use OpenMP for parallel propagation
USE_OPENMP=0

# GNU Scientific Library
GSL_ROOT="" # provide a path if not in system location
