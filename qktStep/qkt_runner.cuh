#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#include "qkt_computation.cuh"
#include "naive_qkt.h" 
#define WARP_SIZE 32




template <int qkv_dim>
__global__ void qkt_kernel_wrapper(half* q, half* k, half* qkt, int b_r, int b_c);




