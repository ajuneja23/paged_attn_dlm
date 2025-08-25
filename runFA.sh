#!/bin/bash
apt-get update
apt-get install cmake -y
cmake -DBUILD_NAIVE_QKT=ON .
cmake --build . --target qktRunner
./qktRunner