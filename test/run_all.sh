#!/bin/bash
./run.sh
for ut in $(find . -name 'ut*'); do ${ut}/run.sh ; done
