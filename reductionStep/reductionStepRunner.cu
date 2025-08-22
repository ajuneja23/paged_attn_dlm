#include "naiveReduction.h"
#include "reductionStepRunner.cuh"


template <int qkv_dim>
__global__ void reductionStepWrapper(float *shared_qkt, float *maxValues,
    float *sumValues, half *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, half *casted_qkt,
   int b_c, int b_r, int kElementsTracked, int qElementsTracked) {
  int warpid = blockIdx.x % WARPS_PER_BLOCK;
  int laneid = threadIdx.x % WARP_SIZE;
  int tid = threadIdx.x;
  reductionStep<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output,
                         intermediateRowMaxes, intermediatePV, casted_qkt,
                         warpid, laneid, tid, b_c, b_r, kElementsTracked,
                         qElementsTracked);
}


int main(int argc, char *argv[]) {
    constexpr int qkv_dim = 64;
    constexpr int b_r=32;
    constexpr int b_c = 32;
    constexpr int kElementsTracked = 32;
    constexpr int qElementsTracked = 32;//let's assume not last block
    float *shared_qkt = new float[b_r * b_c];
    float *maxValues = new float[b_r];
    float *sumValues = new float[b_r];
    float *shared_v = new float[b_c * qkv_dim];
    float *output = new float[b_r * qkv_dim];
    float *intermediateRowMaxes = new float[b_r];
    float *intermediatePV = new float[b_r * qkv_dim];
    float *casted_qkt = new float[b_r * b_c];
    __half *casted_qkt_half = new __half[b_r * b_c];
    __half *shared_v_half = new __half[b_c * qkv_dim];
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
        *d_intermediateRowMaxes, *d_intermediatePV;
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
    cudaMemcpy(d_intermediateRowMaxes, intermediateRowMaxes, b_r * sizeof(float), cudaMemcpyHostToDevice);
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
        d_intermediateRowMaxes, d_intermediatePV, d_casted_qkt, b_c, b_r,
        kElementsTracked, qElementsTracked);
    cudaDeviceSynchronize();
    naive_reduction<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output,
                    intermediateRowMaxes, intermediatePV, casted_qkt, b_c, b_r,
                    kElementsTracked, qElementsTracked);
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