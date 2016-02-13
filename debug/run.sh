#!/bin/bash

# After using this script it is necessary to run again the build.sh script
# for generating again the library with the optimization flags

source ../config.sh
DEBUGFLAGS="-g -O0"

\rm -f exe
\rm -f *.o
\rm -f ../src/*.o
\rm -f ../*.so

#runtime dynamic library path
RPATH="$(pwd)/.."

# Build the library using the debugging flags
cd ../src
   $CC $DEBUGFLAGS -std=c++11 -Wall -Werror -fpic -c *.cpp
   $CC $DEBUGFLAGS -std=c++11 -shared -o lib${LIBNAME}.so *.o $LIBMCI $LIBNFM
   mv lib*.so ../
cd ../debug
echo "Rebuilt the library with the debugging flags"

# Build the debugging main executable
echo "$CC $FLAGS $DEBUGFLAGS -I$(pwd)/../src/ -c *.cpp"
$CC $FLAGS $DEBUGFLAGS -Wall -I$(pwd)/../src/ -c *.cpp
echo "$CC $FLAGS $DEBUGFLAGS -I$(pwd)/../src -L$(pwd)/.. -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME}"
$CC $FLAGS $DEBUGFLAGS -I$(pwd)/../src/ -L$(pwd)/../ -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME}
echo "Rebuilt the debugging executable"
echo ""
echo ""

# Run the debugging executable
valgrind --leak-check=full --track-origins=yes ./exe


