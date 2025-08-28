#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <random>
#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16               // for non fsquare tiles
#define SQUARE_TILE_SIZE TILE_X_SIZE // for 16x16 tiles
#define SHARED_Q_K_DIM TILE_Y_SIZE
#define SHARED_TILE_DIM TILE_Y_SIZE

// using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32



template <int qkv_dim> // step 10 of FA1 paper Algorithm 1
__device__ void initialReductions(float *qkt, __half *casted_qkt, int b_r,
                                  int b_c, int laneid, int warpid, float *maxProposal, float* sumVal) {
  float seenMax = -INFINITY;
  for (int i = laneid; i < b_c; i += WARP_SIZE) {
    seenMax = fmaxf(seenMax, qkt[warpid * b_c + i]);
  }
  __syncthreads();
  __reduce_max_sync(0xFFFFFFFF, seenMax);
  for (int i = laneid; i < b_c; i += WARP_SIZE) {
    qkt[warpid * b_c + i] = qkt[warpid * b_c + i] - seenMax;
    qkt[warpid * b_c + i] = expf(qkt[warpid * b_c + i]);
  }
  __syncwarp();
  float runningSum = 0.0f;
  for (int i = laneid; i < b_c; i += WARP_SIZE) {
    runningSum += qkt[warpid * b_c + i];
  }
  __reduce_add_sync(0xFFFFFFFF, runningSum);
  if (laneid == 0) {
    *maxProposal = seenMax;
    *sumVal = runningSum;
  }
}


template <int qkv_dim>
__device__ void globalSyncReduction()




template <int qkv_dim> //(b_r,b_c) x (b_c,qkv_dim) = (b_r,qkv_dim)
__device__ void calcPVSubroutine(__half *p, __half *v, float *output, int laneid,
                       int warpid, int b_r, int b_c, int *p_uleft, int *v_uleft, float* rC) {
  __half *p_elements = new __half[8];
  __half *v_elements = new __half[4];
  int p_entryCoords[8][2] = {
    {p_uleft[0] + laneid / 4, p_uleft[1] + 2 * (laneid % 4)},
    {p_uleft[0] + laneid / 4, p_uleft[1] + 1 + 2 * (laneid % 4)},
    {p_uleft[0] + laneid / 4, p_uleft[1] + 8 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 9 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 1 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 8 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 9 + 2 * (laneid % 4)},
  };
  for (int i = 0; i < 8; i++) {
    p_elements[i] = p_entryCoords[i][0] * b_c + p_entryCoords[i][1];
  }
  int v_entryCoords[4][2] = {
      {v_uleft[0] + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
      {v_uleft[0] + 1 + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
      {v_uleft[0] + 8 + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
      {v_uleft[0] + 9 + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
  };
  for (int i = 0; i < 4; i++) {
    v_elements[i] = v_entryCoords[i][0] * qkv_dim + v_entryCoords[i][1];
  }
  unsigned const *p_ptr = reinterpret_cast<unsigned const *>(p_elements);
  unsigned const *v_ptr = reinterpret_cast<unsigned const *>(v_elements);
  // use mma instruction
  __syncwarp();
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(rC[0]), "=f"(rC[1]), "=f"(rC[2]), "=f"(rC[3])
               : "r"(p_ptr[0]), "r"(p_ptr[1]), "r"(p_ptr[2]), "r"(p_ptr[3]),
                 "r"(v_ptr[0]), "r"(v_ptr[1]), "f"(rC[0]), "f"(rC[1]),
                 "f"(rC[2]), "f"(rC[3]));
}


template <int qkv_dim>
__device__ void calcPV(__half *p, __half *v, float *output, int laneid,
                       int warpid, int b_r, int b_c) {
  int req_x_tiles = ceilf(qkv_dim / TILE_X_SIZE);
  int req_y_tiles = ceilf(b_r / TILE_Y_SIZE);
  int req_tiles = req_x_tiles * req_y_tiles;
  for (int i = warpid; i < req_tiles; i += WARPS_PER_BLOCK) {
    float rC[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int output_u_left[2] = {(i / req_x_tiles) * TILE_Y_SIZE,
                            (i % req_x_tiles) * TILE_X_SIZE};
    for (int j = 0; j < (b_c / SHARED_TILE_DIM); j++) {
      int p_uleft[2] = {output_u_left[0], j * SHARED_TILE_DIM};
      int v_uleft[2] = {j * SHARED_TILE_DIM, output_u_left[1]};
      calcPVSubroutine<qkv_dim>(p, v, output, laneid, warpid, b_r, b_c, p_uleft,
                                v_uleft, rC);
    }
    output[(output_u_left[0] + laneid / 4) * qkv_dim + output_u_left[1] +
           2 * (laneid % 4)] = rC[0];
    output[(output_u_left[0] + laneid / 4) * qkv_dim + output_u_left[1] +
           2 * (laneid % 4) + 1] = rC[1];
    output[(output_u_left[0] + laneid / 4 + 8) * qkv_dim + output_u_left[1] +
           2 * (laneid % 4)] = rC[2];
    output[(output_u_left[0] + laneid / 4 + 8) * qkv_dim + output_u_left[1] +
           2 * (laneid % 4) + 1] = rC[3];
  }
}

template <int qkv_dim>
__device__ void reductionStep(float *shared_qkt, float *maxValues,
                              float *sumValues, half *shared_v, float *output,
                              float *intermediateRowMaxes,
                              float *intermediateSums,
                              float *intermediatePV, half *casted_qkt,
                              int warpid, int laneid, int tid, int b_c, int b_r,
                              int kElementsTracked, int qElementsTracked) {
  for (int i = warpid; i < qElementsTracked; i += WARPS_PER_BLOCK) {
    initialReductions<qkv_dim>(
        shared_qkt, casted_qkt, b_r, b_c, laneid, warpid, &intermediateRowMaxes[i], &intermediateSums[i]);
    );
  }
  __syncthreads();
   calcPV<qkv_dim>(casted_qkt, shared_v, intermediatePV, laneid, warpid, b_r, b_c);
  __syncthreads();
  // final O_i update
  for (int i = warpid; i < qElementsTracked; i += WARPS_PER_BLOCK) {
    float coefficient = expf(intermediateRowMaxes[i] - maxValues[i]) / (sumValues[i] + 1e-5f);
    for (int j = laneid; j < qkv_dim; j += WARP_SIZE) {
      output[i * qkv_dim + j] += coefficient * intermediatePV[i * qkv_dim + j];
    }
  }
}




