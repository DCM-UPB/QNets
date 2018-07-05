#!/bin/bash

# NOTE:
# Before sourcing this file, please set ROOT_FOLDER to the absolute project root path
# e.g. from a level 1 subfolder: $ export ROOT_FOLDER=$(dirname $(pwd))

# --- You may edit this part according to your needs ---

# Library name
export LIBNAME="ffnn"

# C++ compiler
export CC="g++"

# C++ flags (std=c++11 is necessary)
export FLAGS="-std=c++11 -Wall"

# Optimization flags
export OPTFLAGS="-O3 -flto"

# Profiling flags
export GPERFFLAGS="${OPTFLAGS} -DWITHGPERFTOOLS -lprofiler"

# Debugging flags
export DEBUGFLAGS="-g -O0"

# Coverage flags
export GCOVFLAGS="${DEBUGFLAGS} -fprofile-arcs -ftest-coverage"

# GSL (GNU Scientific Library)
IGSL="-I/usr/include"
LGSL="-L/usr/local/lib"
LIBGSL="-lgsl -lgslcblas"


# ------------- Do not edit the following -------------

# Operating System Name (Linux/Darwin)
export OS_NAME=$(uname)

# External include/library paths and library names
export EXT_I="${IGSL}"
export EXT_L="${LGSL}"
export EXT_LIBS="${LIBGSL}"

# Complete paths, including project paths
export FULL_I="-I${ROOT_FOLDER}/include/ ${EXT_I}"
export FULL_L="-L${ROOT_FOLDER} ${EXT_L}"
export FULL_LIBS="-l${LIBNAME} ${EXT_LIBS}"
