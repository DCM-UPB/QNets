#!/bin/bash
# builds the library with optimization flags

source config.sh
export MYFLAGS="${OPTFLAGS}"
./script/build_common.sh
