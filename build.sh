#!/bin/bash
# builds the library with optimization flags

source config.sh
export MYFLAGS="${OPTFLAGS}"
source build_common.sh
