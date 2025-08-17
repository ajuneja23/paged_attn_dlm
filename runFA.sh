#!/bin/bash
apt-get update
apt-get install cmake -y
mkdir build
mv builder.sh build/
sh build/builder.sh
cd build
./fa1forward 256