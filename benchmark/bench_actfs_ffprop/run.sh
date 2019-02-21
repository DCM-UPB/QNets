#!/bin/sh

bench=bench_actfs_ffprop

cd ../../build/benchmark/
echo
echo "Running benchmark ${bench}..."
./${bench}
echo
