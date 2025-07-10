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

// using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

template <int qkv_dim>
__device__ void reductionStep(float *shared_qkt, float *maxValues,
                              float *sumValues, half *shared_v, float *output,
                              float *intermediateRowMaxes,
                              float *intermediatePV, half *casted_qkt,
                              int warpid, int laneid, int tid, int b_c, int b_r,
                              int kElementsTracked, int qElementsTracked) {
  // calculate maxValues, P_{ij} matrix, and l_ij values. split work for each
  // row across warps
  for (int i = warpid; i < qElementsTracked;
       i += WARPS_PER_BLOCK) { // row in the qk^t matrix
    float m_ijProposal = -INFINITY;
    for (int j = laneid; j < kElementsTracked;
         j += WARP_SIZE) { // col in qk^t matrix
      m_ijProposal = fmaxf(m_ijProposal, shared_qkt[i * b_c + j]);
    }
    __syncwarp();
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      m_ijProposal = fmaxf(m_ijProposal,
                           __shfl_down_sync(0xFFFFFFFF, m_ijProposal, offset));
    }
    if (laneid == 0) {
      maxValues[i] = fmaxf(maxValues[i], m_ijProposal);
    }
    m_ijProposal = __shfl_sync(0xFFFFFFFF, m_ijProposal, 0);
    float runningSum = 0.0f;
    for (int j = laneid; j < b_c; j += WARP_SIZE) {
      shared_qkt[i * b_c + j] -= m_ijProposal;
      if (j >= kElementsTracked) {
        shared_qkt[i * b_c + j] = -INFINITY;
      }
      shared_qkt[i * b_c + j] = expf(shared_qkt[i * b_c + j]);
      runningSum += shared_qkt[i * b_c + j];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      runningSum += __shfl_down_sync(0xFFFFFFFF, runningSum,
                                     offset); // l_{ij} calculation
    }
    runningSum = __shfl_sync(0xFFFFFFFF, runningSum, 0);
    float curMax = maxValues[i];
    maxValues[i] = fmaxf(curMax, m_ijProposal);
    float curRunningSum = sumValues[i]; // m_i
    float l_inew = expf(curMax - fmaxf(curMax, m_ijProposal)) * curRunningSum +
                   expf(m_ijProposal - fmaxf(curMax, m_ijProposal)) *
                       runningSum; // l_i^{new}
    if (laneid == 0) {
      intermediateRowMaxes[i] = m_ijProposal;
    }
    // update O_i
    for (int j = laneid; j < qkv_dim; j += WARP_SIZE) {
      output[i * qkv_dim + j] = (curRunningSum / (l_inew + 1e-5f)) *
                                expf(curMax - fmaxf(curMax, m_ijProposal)) *
                                output[i * qkv_dim + j];
      if (laneid == 0) {
        if (output[i * qkv_dim + j] == 0.0f) {
          printf("output[%d * %d + %d] = %f\n", i, qkv_dim, j,
                 output[i * qkv_dim + j]);
          printf("curRunningSum: %f\n", curRunningSum);
          printf("l_inew: %f\n", l_inew);
          printf("curMax: %f\n", curMax);
          printf("m_ijProposal: %f\n", m_ijProposal);
          __trap();
          return;
        }
      }
      sumValues[i] = l_inew;
    }
    // cast qkt to half
    for (int i = tid; i < b_r * b_c; i += WARP_SIZE * WARPS_PER_BLOCK) {
      casted_qkt[i] = __float2half(shared_qkt[i]);
    }
    __syncthreads();

    // handle p_{ij} by v_j multiplication. p_{ij} is in casted_qkt as a b_r x
    // b_c(16x16 tiling). v_j is shared_v as a b_c x qkv_dim (16x8 tiling)
    int req_x_tiles = ceilf(qkv_dim / TILE_X_SIZE);
    int req_y_tiles = ceilf(b_r / TILE_Y_SIZE);
    int req_tiles = req_x_tiles * req_y_tiles;
    for (int i = warpid; i < req_tiles; i += WARPS_PER_BLOCK) {
      float rC[4] = {0, 0, 0, 0};
      int output_u_left[2] = {
          (i) / req_x_tiles * TILE_Y_SIZE,
          (i) % req_x_tiles *
              TILE_X_SIZE}; // split output tile work across warps
      for (int j = 0; j < (b_c / SHARED_Q_K_DIM); j++) {
        int p_u_left[2] = {output_u_left[0], j * SHARED_Q_K_DIM};
        int v_u_left[2] = {j * SHARED_Q_K_DIM, output_u_left[1]};
        half p_elements[8] = {
            casted_qkt[(p_u_left[0] + laneid / 4) * b_c + p_u_left[1] +
                       2 * (laneid % 4)],
            casted_qkt[(p_u_left[0] + laneid / 4) * b_c + p_u_left[1] +
                       2 * (laneid % 4) + 1],
            casted_qkt[(p_u_left[0] + laneid / 4 + 8) * b_c + p_u_left[1] +
                       2 * (laneid % 4)],
            casted_qkt[(p_u_left[0] + laneid / 4 + 8) * b_c + p_u_left[1] +
                       2 * (laneid % 4) + 1],
            casted_qkt[(p_u_left[0] + laneid / 4) * b_c + p_u_left[1] + 8 +
                       2 * (laneid % 4)],
            casted_qkt[(p_u_left[0] + laneid / 4 + 8) * b_c + p_u_left[1] + 8 +
                       2 * (laneid % 4) + 1],
            casted_qkt[(p_u_left[0] + laneid / 4) * b_c + p_u_left[1] + 8 +
                       2 * (laneid % 4)],
            casted_qkt[(p_u_left[0] + laneid / 4 + 8) * b_c + p_u_left[1] + 8 +
                       2 * (laneid % 4) + 1]};
        half v_elements[4] = {
            shared_v[(v_u_left[0] + 2 * (laneid % 4)) * qkv_dim + v_u_left[1] +
                     laneid / 4],
            shared_v[(v_u_left[0] + 2 * (laneid % 4) + 1) * qkv_dim +
                     v_u_left[1] + laneid / 4],
            shared_v[(v_u_left[0] + 2 * (laneid % 4) + 8) * qkv_dim +
                     v_u_left[1] + laneid / 4],
            shared_v[(v_u_left[0] + 2 * (laneid % 4) + 9) * qkv_dim +
                     v_u_left[1] + laneid / 4]};
        unsigned const *p_ptr = reinterpret_cast<unsigned const *>(p_elements);
        unsigned const *v_ptr = reinterpret_cast<unsigned const *>(v_elements);
        // use mma instruction
        __syncwarp();
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(rC[0]), "=f"(rC[1]), "=f"(rC[2]), "=f"(rC[3])
            : "r"(p_ptr[0]), "r"(p_ptr[1]), "r"(p_ptr[2]), "r"(p_ptr[3]),
              "r"(v_ptr[0]), "r"(v_ptr[1]), "f"(rC[0]), "f"(rC[1]), "f"(rC[2]),
              "f"(rC[3]));
      }
      intermediatePV[(output_u_left[0] + laneid / 4) * qkv_dim +
                     output_u_left[1] + 2 * (laneid % 4)] = rC[0];
      intermediatePV[(output_u_left[0] + laneid / 4) * qkv_dim +
                     output_u_left[1] + 2 * (laneid % 4) + 1] = rC[1];
      intermediatePV[(output_u_left[0] + laneid / 4 + 8) * qkv_dim +
                     output_u_left[1] + 2 * (laneid % 4)] = rC[2];
      intermediatePV[(output_u_left[0] + laneid / 4 + 8) * qkv_dim +
                     output_u_left[1] + 2 * (laneid % 4) + 1] = rC[3];
    }
    __syncthreads();
    // final O_i update
    for (int i = warpid; i < qElementsTracked; i += WARPS_PER_BLOCK) {
      float coefficient =
          expf(intermediateRowMaxes[i] - maxValues[i]) / (sumValues[i] + 1e-5f);
      for (int j = laneid; j < qkv_dim; j += WARP_SIZE) {
        output[i * qkv_dim + j] +=
            coefficient * intermediatePV[i * qkv_dim + j];
        if (laneid == 0) {
          if (output[i * qkv_dim + j] == 0.0f) {
            printf("output[%d * %d + %d] = %f\n", i, qkv_dim, j,
                   output[i * qkv_dim + j]);
            printf("coefficient: %f\n", coefficient);
            printf("intermediateRowMaxes[%d]: %f\n", i,
                   intermediateRowMaxes[i]);
          }
          __trap();
          return;
        }
      }
    }
  }
