#!/bin/bash

# NOTE:
# Before sourcing this file, please set ROOT_FOLDER to the absolute project root path
# e.g. from a level 1 subfolder: $ export ROOT_FOLDER=$(dirname $(pwd))

# --- You may edit this part according to your needs ---

# Library name
LIBNAME="ffnn"

# C++ compiler
CC="g++"

# C++ flags (std=c++11 is necessary)
FLAGS="-std=c++11 -Wall"

# Optimization flags
OPTFLAGS="-O3 -flto"

# Profiling flags
GPERFFLAGS="${OPTFLAGS} -DWITHGPERFTOOLS -lprofiler"

# Debugging flags
DEBUGFLAGS="-g -O0"

# GSL (GNU Scientific Library)
IGSL="-I/usr/include"
LGSL="-L/usr/local/lib"
LIBGSL="-lgsl -lgslcblas"


# ------------- Do not edit the following -------------

# Operating System Name (Linux/Darwin)
OS_NAME=$(uname)

# External include/library paths and library names
EXT_I="${IGSL}"
EXT_L="${LGSL}"
EXT_LIBS="${LIBGSL}"

# Complete paths, including project paths
FULL_I="-I${ROOT_FOLDER}/include/ ${EXT_I}"
FULL_L="-L${ROOT_FOLDER} ${EXT_L}"
FULL_LIBS="-l${LIBNAME} ${EXT_LIBS}"
