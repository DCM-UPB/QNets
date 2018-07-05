#!/bin/bash
# builds the library with coverage flags

source config.sh
export MYFLAGS="${GCOVFLAGS}"
./script/build_common.sh
