#!/bin/bash
# builds the library with set of flags according to argument (opt, debug, profiling)
if [ "$#" -eq 0 ] || [ "$1" == "opt" ]
  then
    ./script/build_opt_library.sh
elif [ "$1" == "debug" ]
  then
    ./script/build_debug_library.sh
elif [ "$1" == "profiling" ]
  then
    ./script/build_profiling_library.sh
else
  echo "Error: Invalid build mode argument!"
fi;
