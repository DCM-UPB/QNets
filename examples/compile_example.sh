#!/bin/bash

export ROOT_FOLDER=$(dirname $(dirname $(pwd)))
source "${ROOT_FOLDER}/config.sh"

FLAG_TO_USE="${OPTFLAGS}"
#FLAG_TO_USE="${DEBUGFLAGS}"

rm -f exe
rm -f *.o

# Build the debugging main executable
echo "$CC $FLAGS $FLAG_TO_USE -Wall ${FULL_I} -c *.cpp"
$CC $FLAGS $FLAG_TO_USE -Wall ${FULL_I} -c *.cpp

# For Mac OS, the install name is wrong and must be corrected
case ${OS_NAME} in
    "Darwin")
        echo "$CC $FLAGS $FLAG_TO_USE ${FULL_L} -o exe *.o ${FULL_LIBS}"
        $CC $FLAGS $FLAG_TO_USE ${FULL_L} -o exe *.o ${FULL_LIBS}
        ;;
    "Linux")
        echo "$CC $FLAGS $FLAG_TO_USE ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}"
        $CC $FLAGS $FLAG_TO_USE ${FULL_L} -Wl,-rpath=${ROOT_FOLDER} -o exe *.o ${FULL_LIBS}
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
