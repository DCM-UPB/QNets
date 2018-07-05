#!/bin/bash

# After using this script it is necessary to run again the build.sh script
# for generating again the library with the optimization flags

rm -f exe
rm -f *.o

# Build the library using the debugging flags
export ROOT_FOLDER=$(dirname $(pwd))
source ../config.sh
cd ..
./build.sh coverage
cd debug
echo "Rebuilt the library with the debugging flags"
echo ""

# Build the debugging main executable
echo "$CC $FLAGS $DEBUGFLAGS ${FULL_I} -c *.cpp"
$CC $FLAGS $DEBUGFLAGS -Wall ${FULL_I} -c *.cpp

case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $DEBUGFLAGS ${FULL_L} -o exe *.o ${FULL_LIBS}"
        $CC $FLAGS $DEBUGFLAGS ${FULL_L} -o exe *.o ${FULL_LIBS}
        ;;
    "Linux")
        echo "$CC $FLAGS $DEBUGFLAGS ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}"
        $CC $FLAGS $DEBUGFLAGS ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}
        ;;
esac

echo "Rebuilt the debugging executable"
echo ""
echo ""

# Run the debugging executable
valgrind --leak-check=full --track-origins=yes ./exe
