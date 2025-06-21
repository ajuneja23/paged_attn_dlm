#include <iostream>
#include "fa1_forward.cuh"
#include <cuda_runtime.h>




int main() {
    int seq_len = 1024;
    int qkv_dim = 1024;
    int num_heads = 16;
    float* d_q;
    float* d_k;
    float* d_v;
    float* d_maxValues;
    float* d_sumValues;
    float* d_output;

    cudaMalloc(&d_q, num_heads * seq_len * qkv_dim * sizeof(float));
    cudaMalloc(&d_k, num_heads * seq_len * qkv_dim * sizeof(float));
    cudaMalloc(&d_v, num_heads * seq_len * qkv_dim * sizeof(float));
    cudaMalloc(&d_maxValues, num_heads * seq_len * sizeof(float));
    cudaMalloc(&d_sumValues, num_heads * seq_len * sizeof(float));
    cudaMalloc(&d_output, num_heads * seq_len * qkv_dim * sizeof(float));
    float* h_q = new float[num_heads * seq_len * qkv_dim];
    float* h_k = new float[num_heads * seq_len * qkv_dim];
    float* h_v = new float[num_heads * seq_len * qkv_dim];
    for (int i = 0; i < num_heads * seq_len * qkv_dim; ++i) {
        h_q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_k[i] = static_cast<float>(rand()) / RAND_MAX;
        h_v[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    float* h_maxValues = new float[num_heads * seq_len];
    float* h_sumValues = new float[num_heads * seq_len];
    for (int i = 0; i < num_heads * seq_len; ++i) {
        h_maxValues[i] = -std::numeric_limits<float>::infinity();
        h_sumValues[i] = 0.0f;
    }
    cudaMemcpy(d_q, h_q, num_heads * seq_len * qkv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, num_heads * seq_len * qkv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, num_heads * seq_len * qkv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxValues, h_maxValues, num_heads * seq_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sumValues, h_sumValues, num_heads * seq_len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8,16);
    dim3 numBlocks((seq_len + threadsPerBlock.x - 1) / threadsPerBlock.x, num_heads);

    fa1_fwd<__half, float, 1024, 16> <<<numBlocks, threadsPerBlock>>>(
        d_q, 
        d_k, 
        d_v, 
        d_maxValues, 
        d_sumValues, 
        d_output, 
        seq_len
    );

    // Copy the result back to host
    float* h_output = new float[num_heads * seq_len * qkv_dim];
    cudaMemcpy(h_output, d_output, num_heads * seq_len * qkv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < num_heads * seq_len * qkv_dim; ++i) {
        std::cout << "output[" << i << "]: " << h_output[i] << std::endl;
    }
    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_maxValues;
    delete[] h_sumValues;
    delete[] h_output;
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_maxValues);
    cudaFree(d_sumValues);
    cudaFree(d_output);
    return 0; 
}
