#!/bin/bash

cd include
rm *.hpp # remove old links
ln -sf ../src/*/*.hpp ./ # make new links to all header files in src
cd ..

cd bin
echo "libffnn_la_HEADERS = \\" > headers.am
find ../src/ -name *.hpp | tr '\n' ' ' >> headers.am

echo "libffnn_la_SOURCES = \\" > sources.am
find ../src/ -name *.cpp | tr '\n' ' ' >> sources.am

cd ..
