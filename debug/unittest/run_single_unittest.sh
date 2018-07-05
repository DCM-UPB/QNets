#!/bin/bash

export ROOT_FOLDER=$(dirname $(dirname $(dirname $(pwd))))
source "${ROOT_FOLDER}/config.sh"

rm -f exe
rm -f *.o

## Build the debugging main executable
$CC $FLAGS $DEBUGFLAGS -Wall ${FULL_I} -c *.cpp

case ${OS_NAME} in
    "Linux")
        $CC $FLAGS $DEBUGFLAGS ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}
        ;;
    "Darwin")
        $CC $FLAGS $DEBUGFLAGS ${FULL_L} -o exe *.o ${FULL_LIBS}
#        install_name_tool -change lib${LIBNAME}.so ${ROOT_FOLDER}/lib${LIBNAME}.so exe
        ;;
    *)
        echo "The detected operating system is not between the known ones (Linux and Darwin)"
        ;;
esac

# Run the debugging executable
./exe

