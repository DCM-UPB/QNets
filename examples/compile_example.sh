#!/bin/bash

source ../../config.sh

FLAG_TO_USE="${OPTFLAGS}"

\rm -f exe
\rm -f *.o

# project root directory
ROOT_FOLDER=$(dirname $(dirname $(pwd)))

#runtime dynamic library path
RPATH="${ROOT_FOLDER}"

# Build the debugging main executable
echo "$CC $FLAGS $FLAG_TO_USE -Wall -I${ROOT_FOLDER}/include/ -c *.cpp"
$CC $FLAGS $FLAG_TO_USE -Wall -I${ROOT_FOLDER}/include/ -c *.cpp

# For Mac OS, the install name is wrong and must be corrected
case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $FLAG_TO_USE -L${ROOT_FOLDER} $LGSL -o exe *.o -l$LIBNAME $LIBGSL"
        $CC $FLAGS $FLAG_TO_USE -L${ROOT_FOLDER} $LGSL -o exe *.o -l$LIBNAME $LIBGSL
        ;;
    "Linux")
        echo "$CC $FLAGS $FLAG_TO_USE $LGSL -L${ROOT_FOLDER} -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME}" $LIBGSL
        $CC $FLAGS $FLAG_TO_USE $LGSL -L${ROOT_FOLDER} -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME} $LIBGSL
        ;;
esac

echo "Rebuilt the executable file"
echo ""
echo ""

# Ready to run the example executable
echo "Ready to run!"
echo ""
echo "--------------------------------------------------------------------------"
echo ""
echo ""
echo ""
