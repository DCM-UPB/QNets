#!/bin/bash
# builds the library with debugging flags

source config.sh
export MYFLAGS="${DEBUGFLAGS}"
./script/build_common.sh
