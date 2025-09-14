#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <mma.h>
#include <random>
#define TILE_X_SIZE 8
#define TILE_Y_SIZE 16               
#define SQUARE_TILE_SIZE TILE_X_SIZE
#define SHARED_Q_K_DIM TILE_Y_SIZE
#define SHARED_TILE_DIM TILE_Y_SIZE

// using ampere m16n8k16 mma
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32



template <int qkv_dim> // step 10 of FA1 paper Algorithm 1
__device__ void initialReductions(float *qkt, half *casted_qkt, int b_r,
                                  int b_c, int laneid, int warpid,
                                  float *maxProposal, float *sumProposal) {
  for (int i = warpid; i < b_r; i += WARPS_PER_BLOCK) { //each warp handles a row
    float seenMax = -INFINITY;
    for (int j = laneid; j < b_c; j += WARP_SIZE) {
      seenMax = fmaxf(seenMax, qkt[i * b_c + j]);
    }
    __syncwarp();
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      seenMax = fmaxf(seenMax, __shfl_down_sync(0xffffffff, seenMax, offset));
    }
    __syncwarp();
    __shfl_sync(0xffffffff, seenMax,
                0); // send to all lanes b/c we want parallelizable max
                    // subtraction and exp step
    float runningSum = 0.0f;
    for (int j = laneid; j < b_c; j += WARP_SIZE) {
      qkt[i * b_c + j] = expf(qkt[i * b_c + j] - seenMax);
      runningSum += qkt[i * b_c + j];
    }
    __syncwarp();
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      runningSum += __shfl_down_sync(0xffffffff, runningSum, offset);
    }
    __syncwarp();
    if (laneid == 0) {
      maxProposal[i] = seenMax;//m_{ij}
      sumProposal[i] = runningSum;//l_{ij}
    }
  }
}


template <int qkv_dim>
__device__ void globalSyncReduction(float *qkt, half *casted_qkt, int b_r,
                                    int b_c, int laneid, int warpid,
                                    float *maxProposal, float *sumProposal,
                                    float *maxValues, float *sumValues, float *intermediateRowMaxes,
                                    float *intermediateSumValues) {
  for (int i = warpid; i < b_r; i += WARPS_PER_BLOCK) {
    if (laneid == 0) {
      intermediateRowMaxes[i] = fmaxf(maxValues[i], maxProposal[i]);//m_i^{new}
    }
    if (laneid == 1) {
      intermediateSumValues[i] =
          expf(maxValues[i] - intermediateRowMaxes[i]) * sumValues[i] +
          expf(maxProposal[i] - intermediateRowMaxes[i]) *
              sumProposal[i]; // l_i^{new}
    }
    __syncwarp();
  }
}




template <int qkv_dim> //(b_r,b_c) x (b_c,qkv_dim) = (b_r,qkv_dim)
__device__ void calcPVSubroutine(half *p, half *v, float *output, int laneid,
                       int warpid, int b_r, int b_c, int *p_uleft, int *v_uleft, float* rC) {
  half p_elements[8];
  half v_elements[4];
  int p_entryCoords[8][2] = {
    {p_uleft[0] + laneid / 4, p_uleft[1] + 2 * (laneid % 4)},
    {p_uleft[0] + laneid / 4, p_uleft[1] + 1 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 1 + 2 * (laneid % 4)},
    {p_uleft[0] + laneid / 4, p_uleft[1] + 8 + 2 * (laneid % 4)},
    {p_uleft[0] + laneid / 4, p_uleft[1] + 9 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 8 + 2 * (laneid % 4)},
    {p_uleft[0] + 8 + laneid / 4, p_uleft[1] + 9 + 2 * (laneid % 4)},
  };
  for (int i = 0; i < 8; i++) {
    p_elements[i] = p[p_entryCoords[i][0] * b_c + p_entryCoords[i][1]];
  }
  int v_entryCoords[4][2] = {
      {v_uleft[0] + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
      {v_uleft[0] + 1 + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
      {v_uleft[0] + 8 + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
      {v_uleft[0] + 9 + 2 * (laneid % 4), v_uleft[1] + laneid / 4},
  };
  for (int i = 0; i < 4; i++) {
    v_elements[i] = v[v_entryCoords[i][0] * qkv_dim + v_entryCoords[i][1]];
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
__device__ void calcPV(half *p, half *v, float *output, int laneid,
                       int warpid, int b_r, int b_c) {
  int req_x_tiles = ceilf(static_cast<float>(qkv_dim) / TILE_X_SIZE);
  int req_y_tiles = ceilf(static_cast<float>(b_r) / TILE_Y_SIZE);
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
    int output_coords[4][2] = {
        {output_u_left[0] + laneid / 4, output_u_left[1] + 2 * (laneid % 4)},
        {output_u_left[0] + laneid / 4, output_u_left[1] + 1 + 2 * (laneid % 4)},
        {output_u_left[0] + laneid / 4 + 8, output_u_left[1] + 2 * (laneid % 4)},
        {output_u_left[0] + laneid / 4 + 8,
         output_u_left[1] + 1 + 2 * (laneid % 4)}};
    for (int idx = 0; idx < 4; idx++) {
      output[output_coords[idx][0] * qkv_dim + output_coords[idx][1]] = rC[idx];
    }
  }
}


__device__ void castQKT(float *qkt, half *casted_qkt, int b_r, int b_c,
                        int laneid, int warpid) {
  int threadid = warpid * WARP_SIZE + laneid;
  for (int i = threadid; i < b_r * b_c; i += WARP_SIZE * WARPS_PER_BLOCK) {
    casted_qkt[i] = __float2half(qkt[i]);
  }
}


template <int qkv_dim>
__device__ void
finalOutputUpdate(float *output, float *intermediatePV, float *sumValues,
                  float *maxValues, float *intermediateSumValues,
                  float *intermediateMaxValues, float *curProposedRowMaxes,
                  float *curProposedSums, int b_r, int b_c, int laneid,
                  int warpid) {
  int threadid = warpid * WARP_SIZE + laneid;
  for (int i = threadid; i < b_r * qkv_dim; i += WARPS_PER_BLOCK * WARP_SIZE) {
    int row = i / qkv_dim;
    float term1 = expf(maxValues[row] - intermediateMaxValues[row]) * output[i];
    float term2 = expf(curProposedRowMaxes[row] - intermediateMaxValues[row]) *
                intermediatePV[i];
    output[i] = (term1 * sumValues[row] + term2) / (1e-5f + intermediateSumValues[row]);
  }
  __syncthreads();
  for (int i = threadid; i < b_r; i += WARPS_PER_BLOCK * WARP_SIZE) {
    sumValues[i] = intermediateSumValues[i];
    maxValues[i] = intermediateMaxValues[i];
  }
}


template <int qkv_dim>
__device__ void
reductionStep(float *shared_qkt, float *maxValues, float *sumValues,
              half *shared_v, float *output, float *intermediateRowMaxes,
              float *intermediateSums, float *curProposedRowMaxes,
              float *curProposedSums, float *intermediatePV, half *casted_qkt,
              int warpid, int laneid, int tid, int b_c, int b_r) {
  initialReductions<qkv_dim>(shared_qkt, casted_qkt, b_r, b_c, laneid, warpid,
                            curProposedRowMaxes, curProposedSums);
  globalSyncReduction<qkv_dim>(shared_qkt, casted_qkt, b_r, b_c, laneid,
                                 warpid, curProposedRowMaxes,
                                 curProposedSums, maxValues,
                                 sumValues, intermediateRowMaxes,
                                  intermediateSums);
  castQKT(shared_qkt, casted_qkt, b_r, b_c, laneid, warpid);
  __syncthreads();
   calcPV<qkv_dim>(casted_qkt, shared_v, intermediatePV, laneid, warpid, b_r, b_c);

   // final O_i update
   finalOutputUpdate<qkv_dim>(output, intermediatePV, sumValues, maxValues,
                              intermediateSums, intermediateRowMaxes,
                              curProposedRowMaxes, curProposedSums, b_r, b_c,
                              laneid, warpid);
   
}




