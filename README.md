[![Build Status](https://travis-ci.com/DCM-UPB/FeedForwardNeuralNetwork.svg?branch=master)](https://travis-ci.com/DCM-UPB/FeedForwardNeuralNetwork)
[![Coverage Status](https://coveralls.io/repos/github/DCM-UPB/FeedForwardNeuralNetwork/badge.svg?branch=master)](https://coveralls.io/github/DCM-UPB/FeedForwardNeuralNetwork?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/259f588d9bd44ca88b9e7dce9f83c36b)](https://www.codacy.com/app/DCM-UPB/FeedForwardNeuralNetwork?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DCM-UPB/FeedForwardNeuralNetwork&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/dcm-upb/feedforwardneuralnetwork/badge)](https://www.codefactor.io/repository/github/dcm-upb/feedforwardneuralnetwork)


# FeedForwardNeuralNetwork

C++ Library for building and using a Feed Forward Neural Network.
It includes first and second derivatives in respect to the input values, and first derivatives in respect to the variational parameters.

To get you started, there is a user manual pdf in `doc/` and in `examples/` there are several basic examples.

Most subdirectories come with a `README.md` file, explaining the purpose and what you need to know.



# Supported Systems

Currently, we automatically test the library on Arch Linux (GCC 8) and MacOS (with clang as well as brewed GCC 8).
However, in principle any system with C++11 supporting compiler should work, at least if you manage to install all dependencies.



# Build the library

Make sure you have a reasonably recent development version (>=2.3?) of the GSL library on your system. Furthermore, we rely on the CMake build system.

Before compiling, copy the config template:
   `cp config_template.sh config.sh`

and edit it as needed (especially if you have the GSL library in non-standard paths.

If you are done, simply use the following script to compile the library and all tests, benchmarks and examples:
   `./build.sh`

Note that we build out-of-tree, so the compiled library and executable files can be found in the directories under `./build/`.

# Multi-threading: OpenMP

This library supports multi-threading computation with a shared memory paradigm, thanks to OpenMP.

To activate this feature, set `USE_OPENMP=1` inside your config.sh, before building. Currently it is not recommended to use this, in most cases.
