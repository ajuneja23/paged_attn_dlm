#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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
__device__ void calcQKT(half *shared_q, half *shared_k, float *shared_qkt,
                        int laneid, int warpid, int b_c, int b_r) {
  int req_x_tiles = ceilf(b_c / TILE_Y_SIZE);
  int req_y_tiles = ceilf(b_r / TILE_X_SIZE);
  int req_tiles =
      req_x_tiles * req_y_tiles; // # of tiles in full qk^t block output
  for (int i = warpid; i < req_tiles; i += WARPS_PER_BLOCK) {

    int x_idx = (i) % req_x_tiles;
    int y_idx = (i) / req_x_tiles;
    int output_tile_uleft[2] = {y_idx * TILE_Y_SIZE,
                                x_idx * TILE_X_SIZE}; // upper left's row, col
    float rC[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < qkv_dim / SHARED_Q_K_DIM; j++) {
      int q_uleft[2] = {output_tile_uleft[0], j * SHARED_Q_K_DIM};
      int k_uleft[2] = {
          output_tile_uleft[1],
          j * SHARED_Q_K_DIM}; // storing transpose directly, row wise traversal
                               // for both Q, K tile
      half q_elements[8] = {
          shared_q[(q_uleft[0] + laneid / 4) * qkv_dim + q_uleft[1] +
                   2 * (laneid % 4)],
          shared_q[(q_uleft[0] + laneid / 4) * qkv_dim + q_uleft[1] +
                   2 * (laneid % 4) + 1],
          shared_q[(q_uleft[0] + laneid / 4 + 8) * qkv_dim + q_uleft[1] +
                   2 * (laneid % 4)],
          shared_q[(q_uleft[0] + laneid / 4 + 8) * qkv_dim + q_uleft[1] +
                   2 * (laneid % 4) + 1],
          shared_q[(q_uleft[0] + laneid / 4) * qkv_dim + q_uleft[1] + 8 +
                   2 * (laneid % 4)],
          shared_q[(q_uleft[0] + laneid / 4) * qkv_dim + q_uleft[1] + 8 +
                   2 * (laneid % 4) + 1],
          shared_q[(q_uleft[0] + laneid / 4 + 8) * qkv_dim + q_uleft[1] + 8 +
                   2 * (laneid % 4)],
          shared_q[(q_uleft[0] + laneid / 4 + 8) * qkv_dim + q_uleft[1] + 8 +
                   2 * (laneid % 4) +
                   1]}; // thank you to https://veitner.bearblog.dev/ for making
                        // the register loading a lot easier
      half k_elements[4] = {
          shared_k[(k_uleft[0] + laneid / 4) * qkv_dim + k_uleft[1] +
                   2 * (laneid % 4)],
          shared_k[(k_uleft[0] + laneid / 4) * qkv_dim + k_uleft[1] +
                   2 * (laneid % 4) + 1],
          shared_k[(k_uleft[0] + laneid / 4) * qkv_dim + k_uleft[1] +
                   2 * (laneid % 4) + 8],
          shared_k[(k_uleft[0] + laneid / 4) * qkv_dim + k_uleft[1] +
                   2 * (laneid % 4) + 9] // danger
      };
      unsigned const *q_ptr = reinterpret_cast<unsigned const *>(
          q_elements); // reinterpret as a 4 element array of unsigned ints
      unsigned const *k_ptr = reinterpret_cast<unsigned const *>(k_elements);

      // use mma instruction
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                   : "=f"(rC[0]), "=f"(rC[1]), "=f"(rC[2]), "=f"(rC[3])
                   : "r"(q_ptr[0]), "r"(q_ptr[1]), "r"(q_ptr[2]), "r"(q_ptr[3]),
                     "r"(k_ptr[0]), "r"(k_ptr[1]), "f"(rC[0]), "f"(rC[1]),
                     "f"(rC[2]), "f"(rC[3]));
    }
    // store to smem
    shared_qkt[(output_tile_uleft[0] + laneid / 4) * b_c +
               output_tile_uleft[1] + 2 * (laneid % 4)] = rC[0];
    shared_qkt[(output_tile_uleft[0] + laneid / 4) * b_c +
               output_tile_uleft[1] + 2 * (laneid % 4) + 1] = rC[1];
    shared_qkt[(output_tile_uleft[0] + laneid / 4 + 8) * b_c +
               output_tile_uleft[1] + 2 * (laneid % 4)] = rC[2];
    shared_qkt[(output_tile_uleft[0] + laneid / 4 + 8) * b_c +
               output_tile_uleft[1] + 2 * (laneid % 4) + 1] = rC[3];
  }
}
