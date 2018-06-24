#!/bin/bash
# This file contains shared code among the different build scripts.
# The calling script needs to set MYFLAGS to the desired compilation flag configuration.

export ROOT_FOLDER=$(pwd)
source config.sh

echo "The Operating System is: "${OS_NAME}  # here we consider only Linux and Darwin (Mac Os X)

# clean
rm -f *.so
mkdir -p bin
cd bin/
rm -f *.o *.so

CPPFILES="${ROOT_FOLDER}/src/*/*.cpp"

echo "$CC $FLAGS $MYFLAGS -fpic ${FULL_I} -c ${CPPFILES}"
$CC $FLAGS $MYFLAGS -fpic ${FULL_I} -c ${CPPFILES}

case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $MYFLAGS -shared -install_name ${ROOT_FOLDER}/lib${LIBNAME}.so -o lib${LIBNAME}.so ${EXT_L} *.o ${EXT_LIBS}"
        $CC $FLAGS $MYFLAGS -shared -install_name ${ROOT_FOLDER}/lib${LIBNAME}.so -o lib${LIBNAME}.so ${EXT_L} *.o ${EXT_LIBS}
        ;;
    "Linux")
        echo "$CC $FLAGS $MYFLAGS -shared -o lib${LIBNAME}.so ${EXT_L} *.o ${EXT_LIBS}"
        $CC $FLAGS $MYFLAGS -shared -o lib${LIBNAME}.so ${EXT_L} *.o ${EXT_LIBS}
        ;;
esac

mv lib${LIBNAME}.so ../lib${LIBNAME}.so
cd ..

echo
echo "Library ready!"
echo
echo "Help, how can I use it?"
echo "1)   $CC -I$(pwd)/include/ -c example.cpp"
echo "     $CC -L$(pwd) example.o -l${LIBNAME}"
echo "2)   $CC -I$(pwd)/include/ -L$(pwd) example.cpp -l${LIBNAME}"
