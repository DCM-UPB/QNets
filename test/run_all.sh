#!/bin/bash
./run.sh
for ut in $(find . -name 'ut*'); do cd $ut && ./run.sh && cd ../ ; done
