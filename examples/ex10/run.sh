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
echo "$CC $FLAGS $FLAG_TO_USE -Wall -I${ROOT_FOLDER}/src/ -I/usr/local/include -c *.cpp"
$CC $FLAGS $FLAG_TO_USE -Wall -I${ROOT_FOLDER}/src/ -I/usr/local/include -c *.cpp

# For Mac OS, the install name is wrong and must be corrected
case ${OS_NAME} in
   "Darwin")
      echo "$CC $FLAGS $FLAG_TO_USE -L${ROOT_FOLDER} $LGSL -o exe *.o -l$LIBNAME $LIBGSL"
      $CC $FLAGS $FLAG_TO_USE -L${ROOT_FOLDER} $LGSL -o exe *.o -l$LIBNAME $LIBGSL
      ;;
   "Linux")
      echo "$CC $FLAGS $FLAG_TO_USE $LGSL -I${ROOT_FOLDER}/src -L${ROOT_FOLDER} -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME}" $LIBGSL
      $CC $FLAGS $FLAG_TO_USE $LGSL -I${ROOT_FOLDER}/src/ -L${ROOT_FOLDER} -Wl,-rpath=${RPATH} -o exe *.o -l${LIBNAME} $LIBGSL
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
python plot.py
#valgrind --leak-check=full --track-origins=yes ./exe