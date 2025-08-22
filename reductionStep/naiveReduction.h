#pragma once 

#include <cmath>
#include <algorithm>



template <int qkv_dim>
void naive_reduction_row_maxes(float *shared_qkt, float *maxValues,
    float *sumValues, float *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, int b_c, int b_r,
    int kElementsTracked, int qElementsTracked, float* curSum);





template <int qkv_dim>
void naive_reduction(float *shared_qkt, float *maxValues,
    float *sumValues, float *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, int b_c, int b_r,
    int kElementsTracked, int qElementsTracked, float* curSum);