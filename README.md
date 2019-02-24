[![Build Status](https://travis-ci.com/DCM-UPB/FeedForwardNeuralNetwork.svg?branch=master)](https://travis-ci.com/DCM-UPB/FeedForwardNeuralNetwork)
[![Coverage Status](https://coveralls.io/repos/github/DCM-UPB/FeedForwardNeuralNetwork/badge.svg?branch=master)](https://coveralls.io/github/DCM-UPB/FeedForwardNeuralNetwork?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/259f588d9bd44ca88b9e7dce9f83c36b)](https://www.codacy.com/app/DCM-UPB/FeedForwardNeuralNetwork?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DCM-UPB/FeedForwardNeuralNetwork&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/dcm-upb/feedforwardneuralnetwork/badge)](https://www.codefactor.io/repository/github/dcm-upb/feedforwardneuralnetwork)


# FeedForwardNeuralNetwork

C++ Library for building and using Feed Forward Neural Networks.
It includes first and second derivatives with respect to the input values, first derivatives with respect to the variational parameters
and mixed derivatives with respect to both input and variational parameters.

To get you started, there is a user manual pdf in `doc/` and in `examples/` there are several basic examples.

In `test/` you can find the unit tests and benchmarking programs in `benchmark`.

Some subdirectories come with an own `README.md` file which provides further information.


# Supported Systems

Currently, we automatically test the library on Arch Linux (GCC 8) and MacOS (with clang as well as brewed GCC 8).
However, in principle any system with C++11 supporting compiler should work.


# Requirements

- CMake, to use our build process
- GNU Scientific Library (~2.3+)
- (optional) OpenMP, to use parallelized propagation (make sure that it is beneficial in your case!)
- (optional) valgrind, to run `./run.sh` in `test/`
- (optional) gperftools, ro run `./run_prof.sh` in `benchmark/`
- (optional) pdflatex, to compile the tex file in `doc/`
- (optional) doxygen, to generate doxygen documentation in `doc/doxygen`


# Build the library

Copy the file `config_template.sh` to `config.sh`, edit it to your liking and then simply execute the command

   `./build.sh`

Note that we build out-of-tree, so the compiled library and executable files can be found in the directories under `./build/`.


# First steps

You may want to read `doc/user_manual.pdf` to get a quick overview of the libraries functionality. However, it is not guaranteed to be perfectly up-to-date and accurate.
Therefore, the best way to get your own code started is by studying the examples in `examples/`. See `examples/README.md` for further guidance.


# Multi-threading: OpenMP

This library supports multi-threading computation with a shared memory paradigm, thanks to OpenMP.

To activate this feature, set `USE_OPENMP=1` inside your config.sh, before building. It is recommended to use this only for larger networks.
You can fine tune performance by setting the `OMP_NUM_THREADS` environment variable.
