#!/bin/bash

cd include
rm *.hpp # remove old links
ln -sf ../src/*/*.hpp ./ # make new links to all header files in src
cd ..
