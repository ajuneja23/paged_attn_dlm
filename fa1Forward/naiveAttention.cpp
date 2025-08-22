#include <cmath>


#include <limits>


#include "naiveAttention.h"
using namespace std;

void naive_attention(float *q, float *key, float *v, float *output, int seq_len,
                     int qkv_dim, int num_heads) {
  float *buffer = new float[seq_len * seq_len];
  for (int i = 0; i < num_heads; i++) {
    // Calculate QK^T and store in buffer
    for (int j = 0; j < seq_len; j++) {
      for (int k = 0; k < seq_len; k++) {
        float dot_product = 0;
        for (int d = 0; d < qkv_dim; d++) {
          dot_product += (q[i * seq_len * qkv_dim + j * qkv_dim + d] *
                          key[i * seq_len * qkv_dim + k * qkv_dim + d]);
        }
        buffer[j * seq_len + k] = dot_product;
      }
    }

    // Apply softmax to each row of the buffer
    for (int j = 0; j < seq_len; j++) {
      float max_val = -std::numeric_limits<float>::infinity();
      for (int k = 0; k < seq_len; k++) {
        if (buffer[j * seq_len + k] > max_val) {
          max_val = buffer[j * seq_len + k];
        }
      }

      float sum_exp = 0;
      for (int k = 0; k < seq_len; k++) {
        buffer[j * seq_len + k] = std::exp(buffer[j * seq_len + k] - max_val);
        sum_exp += buffer[j * seq_len + k];
      }

      for (int k = 0; k < seq_len; k++) {
        buffer[j * seq_len + k] /= sum_exp;
      }
    }

    // Calculate softmax(QK^T)V
    for (int j = 0; j < seq_len; j++) {
      for (int d = 0; d < qkv_dim; d++) {
        float weighted_sum = 0;
        for (int k = 0; k < seq_len; k++) {
          weighted_sum += buffer[j * seq_len + k] *
                          v[i * seq_len * qkv_dim + k * qkv_dim + d];
        }
        output[i * seq_len * qkv_dim + j * qkv_dim + d] = weighted_sum;
      }
    }
  }
}
