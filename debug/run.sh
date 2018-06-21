#!/bin/bash

# After using this script it is necessary to run again the build.sh script
# for generating again the library with the optimization flags

source ../config.sh
OS_NAME=$(uname)

\rm -f exe
\rm -f *.o

#runtime dynamic library path
RPATH="$(dirname $(pwd))"

# Build the library using the debugging flags
cd ..
./build_debug_library.sh
cd debug
echo "Rebuilt the library with the debugging flags"
echo ""

# Build the debugging main executable
echo "$CC $FLAGS $DEBUGFLAGS -I$(pwd)/../include/ -c *.cpp"
$CC $FLAGS $DEBUGFLAGS -Wall -I$(pwd)/../include/ -c *.cpp

case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $DEBUGFLAGS -L$(pwd)/.. -L${RPATH} $LGSL -o exe *.o -l${LIBNAME} $LIBGSL"
        $CC $FLAGS $DEBUGFLAGS -L$(pwd)/../ -L${RPATH} $LGSL -o exe *.o -l${LIBNAME} $LIBGSL
        ;;
    "Linux")
        echo "$CC $FLAGS $DEBUGFLAGS -L$(pwd)/.. -Wl,-rpath=${RPATH} $LGSL -o exe *.o -l${LIBNAME} $LIBGSL"
        $CC $FLAGS $DEBUGFLAGS -L$(pwd)/../ -Wl,-rpath=${RPATH} $LGSL -o exe *.o -l${LIBNAME} $LIBGSL
        ;;
esac

echo "Rebuilt the debugging executable"
echo ""
echo ""

# Run the debugging executable
valgrind --leak-check=full --track-origins=yes ./exe
