[![Build Status](https://travis-ci.com/DCM-UPB/FeedForwardNeuralNetwork.svg?branch=master)](https://travis-ci.com/DCM-UPB/FeedForwardNeuralNetwork)
[![Coverage Status](https://coveralls.io/repos/github/DCM-UPB/FeedForwardNeuralNetwork/badge.svg?branch=master)](https://coveralls.io/github/DCM-UPB/FeedForwardNeuralNetwork?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/259f588d9bd44ca88b9e7dce9f83c36b)](https://www.codacy.com/app/DCM-UPB/FeedForwardNeuralNetwork?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DCM-UPB/FeedForwardNeuralNetwork&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/dcm-upb/feedforwardneuralnetwork/badge)](https://www.codefactor.io/repository/github/dcm-upb/feedforwardneuralnetwork)

# FeedForwardNeuralNetwork

C++ Library for building and using a Feed Forward Neural Network.
It includes first and second derivatives in respect to the input values, and first derivatives in respect to the variational parameters.

In `doc/` there is a user manual in pdf.

In `examples/` there are several examples.



# Build the library

Make sure you have a reasonably recent development version of the GSL library installed on your system. Furthermore, we rely on the Autotools build system and libtool.

If you have the prerequisites on your system, you have to setup the build environment by using the following script in the top level directory:

   `./autogen.sh`

Now you want to configure the build process for your platform by invoking:

   `./configure`

Finally, you are ready to compile all the code files in our repository together, by:

   `make` or `make -jN`

where N is the number of parallel threads used by make. As long as you changed, but didn't remove or add source files, it is sufficient to only run `make` again to rebuild.

If you however removed old or added new code under src, you need to first update the source file lists and include links. Do so by invoking from root folder:

   `script/update_file_lists.sh`

NOTE: All the subdirectories of test, benchmark and examples support calling `make` inside them to recompile local changes.



# Build options

You may enable special compiler flags by using one or more of the following options after `configure`:

   `--enable-debug` : Enables flags (like \-g and \-O0) suitable for debugging

   `--enable-coverage` : Enables flags to generate test coverage reports via gcov

   `--enable-profiling` : Enables flags to generate performance profiles for benchmarks




## Multi-threading: OpenMP

This library supports multi-threading computation with a shared memory paradigm, thanks to OpenMP.

To activate this feature use `--enable-openmp` at configuration. Currently it is not recommended to use this for most cases.
