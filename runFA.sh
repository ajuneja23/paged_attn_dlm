#!/bin/bash
apt-get update
apt-get install cmake -y
cmake -DBUILD_REDUCTION_STEP=ON .
cmake --build . --target reductionStepRunner
./reductionStepRunner