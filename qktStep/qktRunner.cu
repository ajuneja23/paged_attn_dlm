#include "qktRunner.cuh"


template <int qkv_dim>
__global__ void qkt_kernel_wrapper(__half* q, __half* k, float* qkt, int b_r, int b_c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    calcQKT<qkv_dim>(q, k, qkt, lane_id, warp_id, b_c, b_r);
    printf("Debug: qkt last element %f\n", qkt[b_r * b_c - 1]);
}


int main(int argc, char *argv[]) {
  constexpr int qkv_dim = 64;
  constexpr int b_r = 32;
  constexpr int b_c = 32;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  __half *h_q = new __half[b_r * qkv_dim];
  float *cpu_q = new float[b_r * qkv_dim];
  float *cpu_k = new float[b_c * qkv_dim];
  __half *h_k = new __half[b_c * qkv_dim];
  float *h_qkt = new float[b_r * b_c];
  for (int i = 0; i < b_r * qkv_dim; i++) {
    cpu_q[i] = dis(gen);
    h_q[i] = __float2half(cpu_q[i]);
  }
  for (int i = 0; i < b_c * qkv_dim; i++) {
    cpu_k[i] = dis(gen);
    h_k[i] = __float2half(cpu_k[i]);
  }
  for (int i = 0; i < b_r * b_c; i++) {
    h_qkt[i] = 0.0f;
  }
  __half *d_q;
  __half *d_k;
  float *d_qkt;
  cudaMalloc(&d_q, b_r * qkv_dim * sizeof(__half));
  cudaMalloc(&d_k, b_c * qkv_dim * sizeof(__half));
  cudaMalloc(&d_qkt, b_r * b_c * sizeof(float));
  cudaMemcpy(d_q, h_q, b_r * qkv_dim * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k, b_c * qkv_dim * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_qkt, h_qkt, b_r * b_c * sizeof(float), cudaMemcpyHostToDevice);
  dim3 numBlocks(1);
  dim3 threadsPerBlock(WARP_SIZE * 4);

  qkt_kernel_wrapper<qkv_dim><<<numBlocks, threadsPerBlock>>>(d_q, d_k, d_qkt, b_r, b_c);
  cudaDeviceSynchronize();
  cudaMemcpy(h_qkt, d_qkt, b_r * b_c * sizeof(float), cudaMemcpyDeviceToHost);
  // CPU TEST
  float *cpu_qkt = new float[b_r * b_c];
  naive_qkt<qkv_dim>(cpu_q, cpu_k, cpu_qkt, b_r, b_c);
  float allowedError = 1e-1;
  std::cout << "last element from gpu qkt uncasted: " << h_qkt[b_r * b_c - 1] << std::endl;
//   for (int i = 0; i < b_r; i++) {
//     for (int j = 0; j < b_c; j++) {
//       float diff = fabs(h_qkt[i * b_c + j] - cpu_qkt[i * b_c + j]);
//       if (diff > allowedError) {
//         std::cout << "Error at (" << i << "," << j << ")" << std::endl;
//         std::cout << "Device value: " << h_qkt[i * b_c + j] << std::endl;
//         std::cout << "CPU value: " << cpu_qkt[i * b_c + j] << std::endl;
//         std::cout << "Difference: " << diff << std::endl;
//       }
//     }
//   }
  return 0;
}