#!/bin/sh

for bench in bench_*; do
    echo "Running ${bench} ..."
    ./run.sh $bench
done
