#!/bin/sh

if [ "$1" = "" ]; then
  echo "Expected the name of the benchmark to run as first argument."
elif [ "$2" = "" ]; then
  echo "Expected the path of libprofiler.so as second argument."
else
  bench=$1
  lprof=$2
  cd ../build/benchmark/
  echo
  echo "Running benchmark ${bench}..."
  LD_PRELOAD=${lprof} CPUPROFILE=${bench}.prof CPUPROFILE_FREQUENCY=10000 CPUPROFILE_REALTIME=1 ./${bench}
  pprof --text ${bench} ${bench}.prof
  echo
fi
