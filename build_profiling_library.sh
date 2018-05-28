#!/bin/bash
# builds the library with profiling flags

source config.sh
export MYFLAGS="${GPERFFLAGS}"
source build_common.sh
