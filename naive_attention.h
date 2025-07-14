#include <fstream>
#include <iostream>

using namespace std;

void naive_attention(float *q, float *key, float *v, float *output, int seq_len,
                     int qkv_dim, int num_heads);