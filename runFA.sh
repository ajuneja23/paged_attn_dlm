#!/bin/bash
apt-get update
apt-get install cmake -y
cmake -DBUILD_NAIVE_QKT=ON .
cmake --build . --target fa1forward
ls
cd paged_attn_dlm
./fa1forward 256