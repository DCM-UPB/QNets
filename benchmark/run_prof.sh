#!/bin/sh

if [ "$1" = "" ]; then
  echo "Expected the name of the benchmark to run as first argument."
else
  bench=${1%"/"} # remove any trailing /
  lprof=$2
  if [ "$lprof" = "" ]; then
      echo "Using default libprofiler.so path: /usr/lib/libprofiler.so"
      lprof="/usr/lib/libprofiler.so"
  fi;
  cd ../build/benchmark/
  echo
  echo "Running benchmark ${bench}..."
  LD_PRELOAD=${lprof} CPUPROFILE=${bench}.prof CPUPROFILE_FREQUENCY=10000 CPUPROFILE_REALTIME=1 ./${bench}
  pprof --text ${bench} ${bench}.prof
  echo
fi;
