#!/bin/bash

# Config script for custom library and header paths or C++ compiler choice
#
# If you want to edit this template script/config_template.sh,
# copy it over to something like config.sh and edit the gitignored copy.
#

# C++ compiler
export CXX="g++"

# C++ flags
export CXXFLAGS=""

# GSL Library
GSL_L="-L/usr/local/lib"
GSL_I="-I/usr/local/include"


# ! DO NOT EDIT THE FOLLOWING !

# linker flags
export LDFLAGS="${GSL_L}"

# pre-processor flags
export CPPFLAGS="${GSL_I}"
