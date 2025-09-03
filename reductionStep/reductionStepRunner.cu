#include "naiveReduction.h"
#include "reductionStepRunner.cuh"
#include <random>


template <int qkv_dim>
__global__ void
reductionStepWrapper(float *shared_qkt, float *maxValues, float *sumValues,
                     __half *shared_v, float *output,
                     float *intermediateRowMaxes, float *intermediateSums,
                     float *curProposedRowMaxes, float *curProposedSums,
                     float *intermediatePV, __half *casted_qkt, int b_c,
                     int b_r, int kElementsTracked, int qElementsTracked) {
  int tid = threadIdx.x;
  int laneid = tid % WARP_SIZE;
  int warpid = tid / WARP_SIZE;
  reductionStep<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output,
                         intermediateRowMaxes, intermediateSums, curProposedRowMaxes,
                         curProposedSums, intermediatePV, casted_qkt, warpid, laneid,
                         tid, b_c, b_r, kElementsTracked, qElementsTracked);
}

template <int qkv_dim>
__global__ void calcPVWrapper(__half *p, __half *v, float *output, int b_r, int b_c) {
  int laneid = threadIdx.x % WARP_SIZE;
  int warpid = threadIdx.x / WARP_SIZE;
  calcPV<qkv_dim>(p, v, output, laneid, warpid, b_r, b_c);
}


int main(int argc, char *argv[]) {
  char *testType = argv[1];
  if (strcmp(testType, "PV") == 0) {
    constexpr int b_r = 32;
    constexpr int b_c = 32;
    constexpr int qkv_dim = 64;
    constexpr int kElementsTracked = 32;
    constexpr int qElementsTracked = 32;
    float *shared_qkt = new float[b_r * b_c]();
    float *maxValues = new float[b_r]();
    float *sumValues = new float[b_r]();
    float *shared_v = new float[b_c * qkv_dim]();
    float *output = new float[b_r * qkv_dim]();
    float *intermediateRowMaxes = new float[b_r]();
    float *intermediatePV = new float[b_r * qkv_dim]();
    float *curSum = new float[b_r]();
    __half *casted_qkt = new __half[b_r * b_c]();
    __half *shared_v_half = new __half[b_c * qkv_dim]();
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f,1.0f);
    for (int i = 0; i < b_r * b_c; i++) {
      shared_qkt[i] = dis(gen);
      casted_qkt[i] = __float2half(shared_qkt[i]);
    }
    for (int i = 0; i < b_c * qkv_dim; i++) {
      shared_v[i] = dis(gen);
      shared_v_half[i] = __float2half(shared_v[i]);
    }
    __half *d_shared_v, *d_casted_qkt;
    float *d_intermediatePV;
    cudaMalloc(&d_shared_v, b_c * qkv_dim * sizeof(__half));
    cudaMemcpy(d_shared_v, shared_v_half, b_c * qkv_dim * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_casted_qkt, b_r * b_c * sizeof(__half));
    cudaMemcpy(d_casted_qkt, casted_qkt, b_r * b_c * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMalloc(&d_intermediatePV, b_r * qkv_dim * sizeof(float));
    cudaMemcpy(d_intermediatePV, intermediatePV, b_r * qkv_dim * sizeof(float),
               cudaMemcpyHostToDevice);
    dim3 numBlocks(1);
    dim3 threadsPerBlock(WARP_SIZE * WARPS_PER_BLOCK);
    float *gpu_output = new float[b_r * qkv_dim]();
    calcPVWrapper<qkv_dim><<<numBlocks, threadsPerBlock>>>(d_casted_qkt, d_shared_v,
                                                    d_intermediatePV, b_r, b_c);
    cudaMemcpy(gpu_output, d_intermediatePV, b_r * qkv_dim * sizeof(float),
               cudaMemcpyDeviceToHost);
    naive_pv_calculation<qkv_dim>(shared_qkt, shared_v, output, b_c, b_r,
                                  kElementsTracked, qElementsTracked,
                                  intermediatePV);
    float allowedError = 1e-2;
    for (int i = 0; i < b_r * qkv_dim; i++) {
      float diff = fabs(output[i] - gpu_output[i]);
      if (diff > allowedError) {
        std::cout << "Error at " << i << std::endl;
        std::cout << "Device value: " << gpu_output[i] << std::endl;
        std::cout << "CPU value: " << output[i] << std::endl;
        std::cout << "Difference: " << diff << std::endl;
      }
    }
    return 0;
  } 
  constexpr int qkv_dim = 64;
  constexpr int b_r = 32;
  constexpr int b_c = 32;
  constexpr int kElementsTracked = 32;
  constexpr int qElementsTracked = 32;//let's assume not last block
  float *shared_qkt = new float[b_r * b_c]();
  float *maxValues = new float[b_r]();
  float *sumValues = new float[b_r]();
  float *shared_v = new float[b_c * qkv_dim]();
  float *output = new float[b_r * qkv_dim]();
  float *intermediateRowMaxes = new float[b_r]();
  float *intermediateSums = new float[b_r]();
  float *curProposedRowMaxes = new float[b_r]();
  float *curProposedSums = new float[b_r]();
  float *intermediatePV = new float[b_r * qkv_dim]();
  float *casted_qkt = new float[b_r * b_c]();
  __half *casted_qkt_half = new __half[b_r * b_c]();
  __half *shared_v_half = new __half[b_c * qkv_dim]();

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < b_r * b_c; i++) {
    shared_qkt[i] = dis(gen);
    casted_qkt[i] = dis(gen);
    casted_qkt_half[i] = __float2half(casted_qkt[i]);
  }
  for (int i = 0; i < b_r; i++) {
      maxValues[i] = dis(gen);
      sumValues[i] = dis(gen);
      intermediateRowMaxes[i] = dis(gen);
  }
  for (int i = 0; i < b_r * qkv_dim; i++) {
    output[i] = dis(gen);
    intermediatePV[i] = dis(gen);
  }
  for (int i = 0; i < b_c * qkv_dim; i++) {
    shared_v[i] = dis(gen);
    shared_v_half[i] = __float2half(shared_v[i]);
  }
  float *d_shared_qkt, *d_maxValues, *d_sumValues, *d_output,
      *d_intermediateRowMaxes, *d_intermediateSums, *d_curProposedRowMaxes, *d_curProposedSums, *d_intermediatePV;
  __half *d_shared_v, *d_casted_qkt;
  cudaMalloc(&d_shared_qkt, b_r * b_c * sizeof(float));
  cudaMemcpy(d_shared_qkt, shared_qkt, b_r * b_c * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_maxValues, b_r * sizeof(float));
  cudaMemcpy(d_maxValues, maxValues, b_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_sumValues, b_r * sizeof(float));
  cudaMemcpy(d_sumValues, sumValues, b_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_output, b_r * qkv_dim * sizeof(float));
  cudaMemcpy(d_output, output, b_r * qkv_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_intermediateRowMaxes, b_r * sizeof(float));
  cudaMemcpy(d_intermediateRowMaxes, intermediateRowMaxes,
              b_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_intermediateSums, b_r * sizeof(float));
  cudaMemcpy(d_intermediateSums, intermediateSums, b_r * sizeof(float),
              cudaMemcpyHostToDevice);
  cudaMalloc(&d_curProposedRowMaxes, b_r * sizeof(float));
  cudaMemcpy(d_curProposedRowMaxes, curProposedRowMaxes,
              b_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_curProposedSums, b_r * sizeof(float));
  cudaMemcpy(d_curProposedSums, curProposedSums,
              b_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_intermediatePV, b_r * qkv_dim * sizeof(float));
  cudaMemcpy(d_intermediatePV, intermediatePV, b_r * qkv_dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&d_shared_v, b_c * qkv_dim * sizeof(__half));
  cudaMemcpy(d_shared_v, shared_v_half, b_c * qkv_dim * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMalloc(&d_casted_qkt, b_r * b_c * sizeof(__half));
  cudaMemcpy(d_casted_qkt, casted_qkt_half, b_r * b_c * sizeof(__half),
              cudaMemcpyHostToDevice);
  dim3 numBlocks(1);
  dim3 threadsPerBlock(WARP_SIZE * WARPS_PER_BLOCK);
  reductionStepWrapper<qkv_dim><<<numBlocks, threadsPerBlock>>>(
      d_shared_qkt, d_maxValues, d_sumValues, d_shared_v, d_output,
      d_intermediateRowMaxes, d_intermediateSums, d_curProposedRowMaxes, d_curProposedSums, d_intermediatePV, d_casted_qkt, b_c, b_r,
      kElementsTracked, qElementsTracked);
  cudaDeviceSynchronize();
  naive_reduction<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output,
                  intermediateRowMaxes, intermediatePV, b_c, b_r,
                  kElementsTracked, qElementsTracked, curProposedSums);
  float *kernel_output = new float[b_r * qkv_dim];
  cudaMemcpy(kernel_output, d_output, b_r * qkv_dim * sizeof(float),
              cudaMemcpyDeviceToHost);
  float allowedError = 1e-2;
  for (int i = 0; i < b_r; i++) {
    for (int j = 0; j < qkv_dim; j++) {
      float diff = fabs(output[i * qkv_dim + j] - kernel_output[i * qkv_dim + j]);
      if (diff > allowedError) {
        std::cout << "Error at (" << i << "," << j << ")" << std::endl;
        std::cout << "Device value: " << kernel_output[i * qkv_dim + j] << std::endl;
        std::cout << "CPU value: " << output[i * qkv_dim + j] << std::endl;
        std::cout << "Difference: " << diff << std::endl;
      }
    }
  }
}