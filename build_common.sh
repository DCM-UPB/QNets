#!/bin/bash
# This file contains shared code among the different build scripts.
# The calling script needs to set MYFLAGS to the desired compilation flag configuration.

OS_NAME=$(uname)
echo "The Operating System is: "${OS_NAME}  # here we consider only Linux and Darwin (Mac Os X)

source config.sh

\rm -f *.so
cd bin/
\rm -f *.o *.so
echo "$CC $FLAGS $MYFLAGS -fpic -I ../include/ -c ../src/*/*.cpp"
$CC $FLAGS $MYFLAGS -fpic -I ../include/ -c ../src/*/*.cpp

case ${OS_NAME} in
    "Darwin")
        ROOT_FOLDER=$(dirname $(pwd))
        echo "$CC $FLAGS $MYFLAGS -shared -install_name ${ROOT_FOLDER}/lib${LIBNAME}.so -I ../include/ -o lib${LIBNAME}.so *.o"
        $CC $FLAGS $MYFLAGS -shared -install_name ${ROOT_FOLDER}/lib${LIBNAME}.so -I ../include/ -o lib${LIBNAME}.so *.o
        ;;
    "Linux")
        echo "$CC $FLAGS $MYFLAGS -shared -o lib${LIBNAME}.so -I ../include/ *.o"
        $CC $FLAGS $MYFLAGS -shared -o lib${LIBNAME}.so -I ../include/ *.o
        ;;
esac

cp lib${LIBNAME}.so ../
cd ..

echo
echo "Library ready!"
echo
echo "Help, how can I use it?"
echo "1)   $CC -I$(pwd)/include/ -c example.cpp"
echo "     $CC -L$(pwd) example.o -l${LIBNAME}"
echo "2)   $CC -I$(pwd)/include/ -L$(pwd) example.cpp -l${LIBNAME}"
