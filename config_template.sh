#!/bin/bash

OS_NAME=$(uname)

# Library name
LIBNAME="ffnn"

# C++ compiler
CC="g++"

# C++ flags (std=c++11 is necessary)
FLAGS="-std=c++11 -Wall"

# Optimization flags
OPTFLAGS="-O3"

# Profiling flags
GPERFFLAGS="${OPTFLAGS} -DWITHGPERFTOOLS -lprofiler"

# Debuggin flags
DEBUGFLAGS="-g -O0"

# GSL (GNU Scientific Library)
LGSL="-L/usr/local/lib"
LIBGSL="-lgsl -lgslcblas"
