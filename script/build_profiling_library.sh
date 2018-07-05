#!/bin/bash
# builds the library with profiling flags

source config.sh
export MYFLAGS="${GPERFFLAGS}"
./script/build_common.sh
