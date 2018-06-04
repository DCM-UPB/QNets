#!/bin/bash
# builds the library with debugging flags

source config.sh
export MYFLAGS="${DEBUGFLAGS}"
source build_common.sh
