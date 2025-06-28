
#ifndef REDUCTION_STEP_CUH
#define REDUCTION_STEP_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <random>
#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16               // for non square tiles
#define SQUARE_TILE_SIZE TILE_X_SIZE // for 16x16 tiles
#define SHARED_Q_K_DIM TILE_Y_SIZE

// using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

template <int qkv_dim>
__device__ void reductionStep(float *shared_qkt, float *maxValues,
                              float *sumValues, half *shared_v, float *output,
                              float *intermediateRowMaxes,
                              float *intermediatePV, half *casted_qkt,
                              int warpid, int laneid, int tid, int b_c, int b_r,
                              int kElementsTracked, int qElementsTracked);

#endif