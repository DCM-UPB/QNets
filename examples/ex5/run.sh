#!/bin/bash
source ../../config.sh
OS_NAME=$(uname)

\rm -f exe
\rm -f *.o

#runtime dynamic library path
RPATH="$(pwd)/../.."

# Build the main executable
echo "$CC $FLAGS $OPTFLAGS -I$(pwd)/../../src/ -c *.cpp"
$CC $FLAGS $OPTFLAGS -Wall -I$(pwd)/../../src/ -c *.cpp

case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $OPTFLAGS -I$(pwd)/../../src -L$(pwd)/../.. -Wl,-rpath,${RPATH} -o exe *.o -l${LIBNAME}"
        $CC $FLAGS $OPTFLAGS -I$(pwd)/../../src -L$(pwd)/../.. -Wl,-rpath,${RPATH} -o exe *.o -l${LIBNAME}
        ;;
    "Linux")
        echo "$CC $FLAGS $OPTFLAGS -L$(pwd)/../.. -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME}"
        $CC $FLAGS $OPTFLAGS -L$(pwd)/../.. -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME}
        ;;
esac

echo "Rebuilt the executable file"
echo ""
echo ""

# Run the debugging executable
echo "Ready to run!"
echo ""
echo "--------------------------------------------------------------------------"
echo ""
echo ""
echo ""
./exe
