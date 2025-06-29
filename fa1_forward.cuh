#ifndef FA1_FWD_CUH
#define FA1_FWD_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16               // for non square tiles
#define SQUARE_TILE_SIZE TILE_X_SIZE // for 16x16 tiles
#define SHARED_Q_K_DIM TILE_Y_SIZE

// using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32
#include "qkt_computation.cuh"
#include "reductionStep.cuh"

template <typename T> struct shared_mem_requirements {
  int dims[2];
};

template <int qkv_dim, int num_heads>
__global__ void fa1_fwd(half *q, half *k, half *v, float *maxValues,
                        float *sumValues, float *output, int seq_len);

__host__ void fa1_fwd_wrapper(int seq_len);

#endif