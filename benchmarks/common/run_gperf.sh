#!/bin/bash
# Script to compile and run a single main.cpp of a benchmark, with gperftools profiling enabled.
# This script is meant to be sourced by another script which sets the appropriate ROOT_PATH.
# After using this script it is necessary to compile the library again with OPTFLAGS to have the normal optimized version.

# remember current path
MYPATH="$(pwd)"

# Build the library using the profiling flags
echo ""
echo ""
echo "Building the library with profiling flags..."
echo ""
echo ""
cd "${ROOT_PATH}"
./build_profiling_library.sh
cd "${MYPATH}"
echo ""
echo ""
echo "Done."
echo ""
echo ""

source "${ROOT_PATH}/config.sh"

FLAG_TO_USE="${GPERFFLAGS}"

\rm -f exe
\rm -f *.o

#library source path
SRC_PATH="${ROOT_PATH}/src/"

#benchmarks/common path
COM_PATH="${ROOT_PATH}/benchmarks/common/"

echo "Building the executable..."
echo ""
echo ""


# Build the main executable
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

echo ""
echo ""
echo "Done."
echo ""
echo ""

# Run the executable
echo "Running the executable..."
echo ""
echo ""
LD_PRELOAD=/usr/lib/libtcmalloc.so:/usr/lib/libprofiler.so CPUPROFILE=exe.prof ./exe > benchmark_new.out
echo "Done."
echo ""
echo ""
