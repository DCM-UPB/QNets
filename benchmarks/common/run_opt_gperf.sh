#!/bin/bash
# Script to compile and run a single main.cpp of a benchmark, with gperftools profiling enabled
# This script is meant to be sourced by another script which sets the appropriate ROOT_PATH

source "${ROOT_PATH}/config.sh"

FLAG_TO_USE="${OPTFLAGS}"

\rm -f exe
\rm -f *.o

#library source path
SRC_PATH="${ROOT_PATH}/src/"

#benchmarks/common path
COM_PATH="${ROOT_PATH}/benchmarks/common/"

# Build the debugging main executable
echo "$CC $FLAGS $FLAG_TO_USE -Wall -I"${SRC_PATH}" -I/usr/local/include -I"${COM_PATH}" -c *.cpp"
$CC $FLAGS $FLAG_TO_USE -Wall -I"${SRC_PATH}" -I/usr/local/include -I"${COM_PATH}" -c *.cpp

# For Mac OS, the install name is wrong and must be corrected
case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $FLAG_TO_USE -L${ROOT_PATH} $LGSL -o exe *.o -l$LIBNAME $LIBGSL"
        $CC $FLAGS $FLAG_TO_USE -L"${ROOT_PATH}" $LGSL -o exe *.o -l$LIBNAME $LIBGSL
        ;;
    "Linux")
        echo "$CC $FLAGS $FLAG_TO_USE $LGSL -I${SRC_PATH} -L${ROOT_PATH} -Wl,-rpath=${ROOT_PATH} -o exe *.o -l${LIBNAME}" $LIBGSL
        $CC $FLAGS $FLAG_TO_USE $LGSL -I"${SRC_PATH}" -L${ROOT_PATH} -Wl,-rpath="${ROOT_PATH}" -o exe *.o -l${LIBNAME} $LIBGSL
        ;;
esac

echo "Rebuilt the executable file"
echo ""
echo ""

# Run the executable
echo "Ready to run!"
echo ""
echo "--------------------------------------------------------------------------"
echo ""
echo ""
echo ""
LD_PRELOAD=/usr/lib/libtcmalloc.so:/usr/lib/libprofiler.so CPUPROFILE=exe.prof ./exe
