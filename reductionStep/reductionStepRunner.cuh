#pragma once


#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <random>


#include "naiveReduction.h"
#include "reductionStep.cuh"
#define WARP_SIZE 32



template <int qkv_dim>
__global__ void
reductionStepWrapper(float *shared_qkt, float *maxValues, float *sumValues,
                     half *shared_v, float *output, float *intermediateRowMaxes,
                     float *intermediatePV, half *casted_qkt, int b_c, int b_r);