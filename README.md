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

Make sure you have a reasonably recent development version (>=2.3?) of the GSL library installed on your system. Furthermore, we rely on the Autotools build system and libtool.
Optionally, if you have valgrind installed on your system, it will be used to check for memory errors when running unittests.

If you have the prerequisites on your system, you have to setup the build environment by using the following script in the top level directory:

   `./autogen.sh`

Now you want to configure the build process for your platform by invoking:

   `./configure`

If you run into trouble here, `./configure --help` could be a start.

Finally, you are ready to compile all the code files in our repository together, by:

   `make` or `make -jN`

where N is the number of parallel threads used by make. Alternatively, you may use the following make targets to build only subparts of the project:

   `make lib`, `make test`, `make benchmark`, `make examples`


As long as you changed, but didn't remove or add source files, it is sufficient to only run `make` again to rebuild.

If you however removed old or added new code files under `src/`, you need to first update the source file lists and include links. Do so by invoking from root folder:

   `make update-sources`

NOTE: All the subdirectories of test, benchmark and examples support calling `make` inside them to recompile local changes.



# Installation

To install the freshly built library and headers into the standard system paths, run (usually sudo is required):
  `make install`

If you however want to install the library under a custom path, before installing you have to use
  `./configure --prefix=/your/absolute/path



# Build options

You may enable special compiler flags by using one or more of the following options after `configure`:

   `--enable-debug` : Enables flags (like \-g and \-O0) suitable for debugging

   `--enable-coverage` : Enables flags to generate test coverage reports via gcov

   `--enable-profiling` : Enables flags to generate performance profiles for benchmarks




## Multi-threading: OpenMP

This library supports multi-threading computation with a shared memory paradigm, thanks to OpenMP.

To activate this feature use `--enable-openmp` at configuration. Currently it is not recommended to use this for most cases.
