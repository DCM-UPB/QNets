#!/bin/bash
source ../compile_example.sh
./exe
#valgrind --leak-check=full --track-origins=yes ./exe
