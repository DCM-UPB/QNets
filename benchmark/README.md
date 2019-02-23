# Benchmarks

This directory contains benchmarks to test the performance of certain parts of the library.
The `common` subfolder contains common source code and script files that are used by the individual benchmarks in `bench_*` folders.

A conda environment that can be used to run the python script is available as `conda-env.yml`

Currently there are the following benchmarks:

   `bench_actfs_deriv`: Benchmark of activation function derivative calculation for various activation functions.

   `bench_actfs_ffprop`: Benchmark of a FFNN's propagation for various hidden layer activation functions.

   `bench_nunits_ffprop`: Benchmark of a FFNN's propagation for different sizes of input and hidden layers.


# Using the benchmarks

Just provide the script `run.sh` the desired benchmark's name, e.g.:
   `./run.sh bench_actfs_ffprop`

Alternatively, you can run all benchmarks sequentially by calling:
   `./run_all.sh`

The benchmark results will be written to a file named `benchmark_new.out` under the respective benchmark folder.
You may visualize the result by entering that directory and using:
   `python plot.py benchmark_new.out`

To let the plot compare the new result versus an older one, you have to provide the old output file like:
   `python plot.py benchmark_old.out benchmark_new.out`.

You may also change new/old to more meaningful labels, anything like benchmark_*.out is allowed (except extra _ or . characters). The
provided labels will be used automatically to create the plot legends.


# Profiling

If you want to performance profile the library under execution of a benchmark,
you just need to provide gperftools's libprofiler.so library to `run_prof.sh` as second argument, e.g.:
   `./run_prof.sh bench_actfs_ffprop /usr/lib/libprofiler.so`

Note that this script does not save any benchmark results.
Also note that for profiling you might want to avoid LTO flags when building the library, to avoid cryptic LTO chunk names in the profile.
