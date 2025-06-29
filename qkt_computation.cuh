#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mma.h>
#include <random>

#ifndef QKT_COMPUTATION_CUH
#define QKT_COMPUTATION_CUH

#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16               // for non square tiles
#define SQUARE_TILE_SIZE TILE_X_SIZE // for 16x16 tiles
#define SHARED_Q_K_DIM TILE_Y_SIZE

// using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

template <int qkv_dim>
__device__ void calcQKT(half *shared_q, half *shared_k, float *shared_qkt,
                        int laneid, int warpid, int b_c, int b_r);
#endif