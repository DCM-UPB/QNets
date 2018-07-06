#!/bin/bash

# Runs the debug compile test and then all unittests
# NOTE: After using this script it is necessary to run again the regular 
# build.sh script for generating again the library with the optimization flags

./run.sh && ./run_tests.sh
