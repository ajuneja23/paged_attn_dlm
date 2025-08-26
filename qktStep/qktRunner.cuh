#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cmath>
#include <mma.h>
#include <random>


#include "qktComputation.cuh"
#include "naiveQkt.h" 
#define WARP_SIZE 32




template <int qkv_dim>
__global__ void qkt_kernel_wrapper(__half* q, __half* k, __half* qkt, int b_r, int b_c);




