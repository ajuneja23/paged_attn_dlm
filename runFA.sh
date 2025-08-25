#!/bin/bash
apt-get update
apt-get install cmake -y
cmake BUILD_NAIVE_QKT .
ls
cd paged_attn_dlm
./fa1forward 256