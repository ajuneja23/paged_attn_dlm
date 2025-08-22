#!/bin/bash
apt-get update
apt-get install cmake -y
cmake .
cmake --build . --config release
./fa1forward 256