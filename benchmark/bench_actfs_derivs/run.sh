#!/bin/sh

bench=bench_actfs_derivs

cd ../../build/benchmark/
echo
echo "Running benchmark ${bench}..."
./${bench}
echo
