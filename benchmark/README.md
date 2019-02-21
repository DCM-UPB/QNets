# Benchmarks

This directory contains benchmarks to test the performance of certain parts of the library.
The `common` subfolder contains common source code and script files that are used by the individual benchmarks in `bench_*` folders.

A conda environment that can be used to run the python script is available as `conda-env.yml`

Currently there are the following benchmarks:

   `bench_actfs_deriv`: Benchmark of activation function derivative calculation for various activation functions.

   `bench_actfs_ffprop`: Benchmark of a FFNN's propagation for various hidden layer activation functions.


# Using the benchmarks

Enter the desired benchmark's directory and execute:
   `./run.sh`

Each benchmark will write the results to the command line output by default.
If you save it into a file `benchmark_new.out` instead (e.g. via `./run.sh > benchmark_new.out`), you may visualize the result by using:
   `python plot.py benchmark_new.out`

To let the plot compare the new result versus an older one, you have to provide the old output file like:
   `python plot.py benchmark_old.out benchmark_new.out`.

You may also change new/old to more meaningful labels, anything like benchmark_*.out is allowed (except extra _ or . characters). The
provided labels will be used automatically to create the plot legends.


# Profiling (currently unavailable)

If you want to use the benchmarks for profiling, recompile the library and benchmarks after configuring
   `./configure --enable-profiling`

Then execute a benchmark via make (!) and afterwards view the profile with:
   `pprof --text exe exe.prof`
