# Benchmarks

This directory contains benchmarks to test the performance of certain parts of the library.
The `common` subfolder contains common source code and script files that are used by the individual benchmarks in `bench_*` folders.

Currently there are the following benchmarks:

   `bench_actfs_deriv`: Benchmark of activation function derivative calculation for various activation functions.

   `bench_actfs_ffprop`: Benchmark of a FFNN's propagation for various hidden layer activation functions.


# Using the benchmarks

Enter the desired benchmark's directory and execute:
   `./run.sh`

Each benchmark will write the result into a file `benchmark_new.out`. For visualization execute the plot script:
   `python plot.py benchmark_new.out`

To let the plot compare the new result versus an older one, you have to provide the old output file like:
   `python plot.py benchmark_old.out benchmark_new.out`.

You may also change new/old to more meaningful labels, anything like benchmark_*.out is allowed (except extra _ or . characters).

# Profiling

If you want to use the benchmarks for profiling, instead execute:
   `./run_gperf.sh`

And then view the profile with:
   `pprof --text exe exe.prof`
