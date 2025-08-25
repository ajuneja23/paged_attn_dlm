#!/bin/bash
apt-get update
apt-get install cmake -y
cmake BUILD_NAIVE_QKT --build . --config release
./fa1forward 256