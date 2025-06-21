#ifndef FA1_FWD_CUH
#define FA1_FWD_CUH



#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <cmath>
#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16//for non square tiles 
#define SQUARE_TILE_SIZE TILE_X_SIZE//for 16x16 tiles
#define SHARED_Q_K_DIM TILE_Y_SIZE

//using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32





template <typename T1, typename T2,int b_c, int b_r, int qkv_dim>
__device__ void calcQKT(T1* shared_q, T1* shared_k, T2* shared_qkt,int seq_len, int laneid,int warpid);


template<typename T1, typename T2, int b_c, int b_r>
__device__ void reductionStep(T2* shared_qkt, T2* maxValues, T2* sumValues,T2* output, T2* intermediateRowMaxes, T2* intermediatePV, T1* casted_qkt, int warpid, int laneid);



template<typename T1, typename T2,int qkv_dim, int num_heads>
__global__ void fa1_fwd(T1* q, T1* k, T1* v, T2* maxValues, T2* sumValues, T2* output,int seq_len);


#endif