#pragma once


#include <cmath>

template <int qkv_dim>
void naive_qkt(float* q, float* k, float* qkt, int b_r, int b_c);