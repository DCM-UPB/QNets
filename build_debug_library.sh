#!/bin/bash

source config.sh

\rm -f *.so
cd src/
   \rm -f *.o *.so
   echo "$CC $FLAGS $DEBUGFLAGS -fpic -c *.cpp"
   $CC $FLAGS $DEBUGFLAGS -fpic -c *.cpp
   echo "$CC $FLAGS $DEBUGFLAGS -shared -o lib${LIBNAME}.so *.o"
   $CC $FLAGS $DEBUGFLAGS -shared -o lib${LIBNAME}.so *.o
   mv lib${LIBNAME}.so ../
cd ..

echo
echo "Library ready!"
echo
echo "Help, how can I use it?"
echo "1)   $CC -I$(pwd)/src/ -c example.cpp"
echo "     $CC -L$(pwd) example.o -l${LIBNAME}"
echo "2)   $CC -I$(pwd)/src/ -L$(pwd) example.cpp -l${LIBNAME}"
