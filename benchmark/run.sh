#!/bin/sh

if [ "$1" = "" ]; then
  echo "Expected the name of the benchmark to run as first argument."
else
  bench=$1
  outfile="$(pwd)/${bench}/benchmark_new.out"
  cd ../build/benchmark/
  echo
  echo "Running benchmark ${bench}..."
  ./${bench} > ${outfile}
  cat ${outfile}
  echo
fi
