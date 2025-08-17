//https://github.com/ajuneja23/paged_attn_dlm.git

#include "fa1_forward.cuh"
#include <iostream>
// parallelize on heads first
template <int qkv_dim, int num_heads>
__global__ void
fa1_fwd(half *q, half *k, half *v, float *maxValues, float *sumValues,
        float *output, int seq_len, int b_c, int b_r,
        int *sizePrefixes) { // q layout is (qkv_dim,seq_len,num_heads):
                             // (1, qkv_dim,qkv_dim*seq_len). same for k,v
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  int bid = blockIdx.x;

  // shared_q[b_r][qkv_dim];
  // shared_k[b_c][qkv_dim];
  //  shared_v[b_c][qkv_dim];
  //  shared_maxValues[b_r];
  //  shared_sumValues[b_r];
  //  shared_output[b_r][qkv_dim];
  //  shared_qkt[b_r][b_c];
  //  shared_intermediateRowMaxes[b_r];
  //  casted_qkt[b_r][b_c];
  //  shared_intermediatePV[b_r][qkv_dim];//need to
  // combine all of this into one shmem
  extern __shared__ char shared_mem[];

  half *shared_q = reinterpret_cast<half *>(shared_mem + sizePrefixes[0]);
  half *shared_k = reinterpret_cast<half *>(shared_mem + sizePrefixes[1]);
  half *shared_v = reinterpret_cast<half *>(shared_mem + sizePrefixes[2]);
  half *shared_casted_qkt =
      reinterpret_cast<half *>(shared_mem + sizePrefixes[3]);
  float *shared_maxValues =
      reinterpret_cast<float *>(shared_mem + sizePrefixes[4]);
  float *shared_sumValues =
      reinterpret_cast<float *>(shared_mem + sizePrefixes[5]);
  float *shared_output =
      reinterpret_cast<float *>(shared_mem + sizePrefixes[6]);
  float *shared_qkt = reinterpret_cast<float *>(shared_mem + sizePrefixes[7]);
  float *shared_intermediateRowMaxes =
      reinterpret_cast<float *>(shared_mem + sizePrefixes[8]);
  float *shared_intermediatePV =
      reinterpret_cast<float *>(shared_mem + sizePrefixes[9]);
  int warpid = tid / WARP_SIZE;
  int laneid = tid % WARP_SIZE;

  int head_id = bid;
  if (bid < num_heads) { // bid=head_id
    int head_prefix = head_id * seq_len * qkv_dim;
    int t_c = ceilf(static_cast<float>(seq_len) / b_c);
    int t_r = ceilf(static_cast<float>(seq_len) / b_r);
    for (int j = 0; j < t_c; j++) { // load in qkv_dim*b_c elements
      int elementsToLoad = b_c * qkv_dim;
      int trueElementsToLoad = -1;
      int kElementsTracked = b_c;

      if (j == t_c - 1) {
        kElementsTracked = seq_len - j * b_c;
        trueElementsToLoad = seq_len * qkv_dim - j * b_c * qkv_dim;
      }
      int seq_prefix = j * b_c * qkv_dim;
      for (int z = tid; z < elementsToLoad;
           z += (WARP_SIZE * WARPS_PER_BLOCK)) {
        if (j == t_c - 1 && z >= trueElementsToLoad) {
          shared_k[z] = 0;
          shared_v[z] = 0;
        }
        else {
          shared_k[z] = k[head_prefix + seq_prefix + z]; // LOAD SHARED K,V
          shared_v[z] = v[head_prefix + seq_prefix + z];
        }
      }
      __syncthreads();
      for (int i = 0; i < t_r; i++) {
        trueElementsToLoad = -1;
        int qElementsTracked = b_r;
        if (i == t_r - 1) {
          qElementsTracked = seq_len - i * b_r;
          trueElementsToLoad = qElementsTracked*qkv_dim;
        }
        int q_prefix = i * b_r * qkv_dim;
        int elementsToLoad = b_r * qkv_dim;
        for (int z = tid; z < elementsToLoad;
             z += (WARP_SIZE * WARPS_PER_BLOCK)) {
          if (i == t_r - 1 && z >= trueElementsToLoad) {
            shared_q[z] = 0;
          }
          else {
            shared_q[z] = q[head_prefix + q_prefix + z];
          }
        }

        // load in maxValues, sumValues
        __syncthreads();
        calcQKT<qkv_dim>(shared_q, shared_k, shared_qkt, laneid, warpid, b_c,
                         b_r);
        __syncthreads();
        //  load in all required sram utils from dram
        //  first half of warps load in maxValues, second half load in
        //  sumValues
        if (warpid < WARPS_PER_BLOCK / 2) {
          for (int z = tid; z < qElementsTracked;
               z += (WARP_SIZE * WARPS_PER_BLOCK / 2)) {
            shared_maxValues[z] = maxValues[head_id * seq_len + i * b_r + z];
          }
        } else {
          for (int z = tid - (WARP_SIZE * WARPS_PER_BLOCK / 2);
               z < qElementsTracked; z += (WARP_SIZE * WARPS_PER_BLOCK / 2)) {
            shared_sumValues[z] = sumValues[head_id * seq_len + i * b_r + z];
          }
        }
        // collaborate on O block loading
        for (int z = tid; z < qElementsTracked * qkv_dim;
             z += (WARP_SIZE * WARPS_PER_BLOCK)) {
          shared_output[z] = output[head_prefix + i * b_r * qkv_dim + z];
        }
        __syncthreads();
        reductionStep<qkv_dim>(
            shared_qkt, shared_maxValues, shared_sumValues, shared_v,
            shared_output, shared_intermediateRowMaxes, shared_intermediatePV,
            shared_casted_qkt, warpid, laneid, tid, b_c, b_r, kElementsTracked,
            qElementsTracked);
        __syncthreads();
        // write output to DRAM
        if (warpid < WARPS_PER_BLOCK / 2) {
          for (int z = tid; z < qElementsTracked;
               z += (WARP_SIZE * WARPS_PER_BLOCK / 2)) {
            maxValues[head_id * seq_len + i * b_r + z] = shared_maxValues[z];
          }
        } else {
          for (int z = tid - (WARP_SIZE * WARPS_PER_BLOCK / 2);
               z < qElementsTracked; z += (WARP_SIZE * WARPS_PER_BLOCK / 2)) {
            sumValues[head_id * seq_len + i * b_r + z] = shared_sumValues[z];
          }
        }
        // collaborate on O block loading
        for (int z = tid; z < qElementsTracked * qkv_dim;
             z += (WARP_SIZE * WARPS_PER_BLOCK)) {
          output[head_prefix + b_r * i * qkv_dim + z] =
              shared_output[z]; // layout is (head_id,token_id,component_id)
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <sequence_length>" << std::endl;
    return 1;
  }
  int seq_len = std::stoi(argv[1]);
  std::cout << "sequence length: " << seq_len << std::endl;
  constexpr int qkv_dim = 64;
  constexpr int num_heads = 1;
  __half *d_q;
  __half *d_k;
  __half *d_v;
  float *d_maxValues;
  float *d_sumValues;
  float *d_output;
  int b_c = 32;
  int b_r = 32;
  cudaMalloc(&d_q, num_heads * seq_len * qkv_dim * sizeof(__half));
  cudaMalloc(&d_k, num_heads * seq_len * qkv_dim * sizeof(__half));
  cudaMalloc(&d_v, num_heads * seq_len * qkv_dim * sizeof(__half));
  cudaMalloc(&d_maxValues, num_heads * seq_len * sizeof(float));
  cudaMalloc(&d_sumValues, num_heads * seq_len * sizeof(float));
  cudaMalloc(&d_output, num_heads * seq_len * qkv_dim * sizeof(float));
  std::cout << "allocated memory on device!" << std::endl;
  __half *h_q = new __half[num_heads * seq_len * qkv_dim];
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  float *float_h_q = new float[num_heads * seq_len * qkv_dim];
  float *float_h_k = new float[num_heads * seq_len * qkv_dim];
  float *float_h_v = new float[num_heads * seq_len * qkv_dim];
  __half *h_k = new __half[num_heads * seq_len * qkv_dim];
  __half *h_v = new __half[num_heads * seq_len * qkv_dim];
  for (int i = 0; i < num_heads * seq_len * qkv_dim; ++i) {
    float_h_q[i] = dis(gen);
    float_h_k[i] = dis(gen);
    float_h_v[i] = dis(gen);
    h_q[i] = __float2half(float_h_q[i]);
    h_k[i] = __float2half(float_h_k[i]);
    h_v[i] = __float2half(float_h_v[i]);
  }
  float *h_maxValues = new float[num_heads * seq_len];
  float *h_sumValues = new float[num_heads * seq_len];
  float *h_output = new float[num_heads * seq_len * qkv_dim];
  for (int i = 0; i < num_heads * seq_len; ++i) {
    h_maxValues[i] = -INFINITY;
    h_sumValues[i] = 0.0f;
    for (int j = 0; j < qkv_dim; ++j) {
      h_output[i * qkv_dim + j] = 0.0f; // initialize output to 0
    }
  }
  std::cout << "filled host memory!" << std::endl;
  cudaMemcpy(d_q, h_q, num_heads * seq_len * qkv_dim * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k, num_heads * seq_len * qkv_dim * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, num_heads * seq_len * qkv_dim * sizeof(__half),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_maxValues, h_maxValues, num_heads * seq_len * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_sumValues, h_sumValues, num_heads * seq_len * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, num_heads * seq_len * qkv_dim * sizeof(float),
             cudaMemcpyHostToDevice);
  dim3 threadsPerBlock(128);
  dim3 numBlocks(1);//one for each head
  // calc shmem size
  shared_mem_requirements<__half> halfshmem_req[4] = {
      {.dims = {b_r, qkv_dim}}, // q
      {.dims = {b_c, qkv_dim}}, // k
      {.dims = {b_c, qkv_dim}}, // v
      {.dims = {b_r, b_c}},     // casted_qkt
  };
  shared_mem_requirements<float> floatshmem_req[6] = {
      {.dims = {b_r, 1}},       {.dims = {b_r, 1}},
      {.dims = {b_r, qkv_dim}}, // maxValues, sumValues,
      {.dims = {b_r, b_c}},     {.dims = {b_r, 1}},
      {.dims = {b_r, qkv_dim}}};
  int total_size = 0;
  int sizePrefixes[10] = {0};
  for (int i = 0; i < 4; i++) {
    int byteCount =
        halfshmem_req[i].dims[0] * halfshmem_req[i].dims[1] * sizeof(__half);
    total_size += byteCount;
    sizePrefixes[i + 1] = sizePrefixes[i] + byteCount;
  }
  for (int i = 4; i < 10; i++) {
    int byteCount = floatshmem_req[i - 4].dims[0] *
                    floatshmem_req[i - 4].dims[1] * sizeof(float);
    total_size += byteCount;
    sizePrefixes[i + 1] = sizePrefixes[i] + byteCount;
  }
  int *d_sizePrefixes;
  cudaMalloc(&d_sizePrefixes, sizeof(int) * 10);
  cudaMemcpy(d_sizePrefixes, sizePrefixes, sizeof(int) * 10,
             cudaMemcpyHostToDevice);
  // std::cout << "total shmem size: " << total_size << std::endl;
  // std::cout << "size prefixes: ";
  for (int i = 0; i < 10; i++) {
    // std::cout << sizePrefixes[i] << " ";
  }
  // std::cout << std::endl;
  std::cout << "starting kernel!" << std::endl;
  fa1_fwd<qkv_dim, num_heads><<<numBlocks, threadsPerBlock, total_size>>>(
      d_q, d_k, d_v, d_maxValues, d_sumValues, d_output, seq_len, b_c, b_r,
      d_sizePrefixes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  cudaMemcpy(h_output, d_output, num_heads * seq_len * qkv_dim * sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "copied result to host!" << std::endl;
  // return 0;
  // for (int i = 0; i < num_heads * seq_len * qkv_dim; ++i) {
  //   std::cout << "h_output[" << i << "]: " << h_output[i] << std::endl;
  // }
  // CPU ATTENTION CHECK
  constexpr float err_tolerance = 1e-1f;
  float *output_cpu = new float[num_heads * seq_len * qkv_dim];
  naive_attention(float_h_q, float_h_k, float_h_v, output_cpu, seq_len, qkv_dim,
                  num_heads);
  // for (int i = 0; i < num_heads * seq_len * qkv_dim; ++i) {
  //   std::cout << "output_cpu[" << i << "]: " << output_cpu[i] << std::endl;
  // }
  for (int i = 0; i < num_heads * seq_len * qkv_dim; ++i) {
    if (abs(h_output[i] - output_cpu[i]) > err_tolerance) {
      printf("error encountered!!!!!\n");
      std::cout << "h_output[" << i << "]: " << h_output[i] << " != output_cpu["  << i << "]: " << output_cpu[i] << std::endl;
    }
  }

  delete[] h_q;
  delete[] h_k;
  delete[] h_v;
  delete[] h_maxValues;
  delete[] h_sumValues;
  delete[] h_output;
  delete[] float_h_q;
  delete[] float_h_k;
  delete[] float_h_v;
  delete[] output_cpu;
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_maxValues);
  cudaFree(d_sumValues);
  cudaFree(d_output);
  std::cout << "freed memory on device!" << std::endl;
}
