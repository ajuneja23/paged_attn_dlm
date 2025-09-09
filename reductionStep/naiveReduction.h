#pragma once 

#include <cmath>
#include <algorithm>



template <int qkv_dim>
void naive_reduction_row_maxes(float *shared_qkt, float *maxValues,
    float *sumValues, float *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, int b_c, int b_r, float* curSum) {
        //calc m_{ij} 
        for (int i = 0; i < b_r; i++) {
            intermediateRowMaxes[i] = -INFINITY;
            for (int j = 0;j < b_c; j++) {
                intermediateRowMaxes[i] = std::max(intermediateRowMaxes[i],shared_qkt[i * b_c + j]);
            }
        }
        //calc s_{ij} and l_{ij}
        for (int i = 0;i < b_r;i++) {
            curSum[i] = 0.0f;
            for (int j = 0; j < b_c; j++) {
                shared_qkt[i * b_c + j] = expf(shared_qkt[i * b_c + j] - intermediateRowMaxes[i]);
                curSum[i] += shared_qkt[i * b_c + j];
            }
        }
    }



template <int qkv_dim> //(b_r,b_c) x (b_c,qkv_dim) is (b_r,qkv_dim)
void naive_pv_calculation(float *shared_qkt, float *shared_v,
                            float *intermediatePV, int b_c, int b_r) {
    for (int i = 0; i < b_r; i++) {
        for (int j = 0; j < qkv_dim; j++) {
            for (int k = 0; k < b_c; k++) {
                intermediatePV[i * qkv_dim + j] += shared_qkt[i * b_c + k] * shared_v[k * qkv_dim + j];
            }
        }
    }
}


template <int qkv_dim>
void naive_reduction(float *shared_qkt, float *maxValues,
    float *sumValues, float *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, int b_c, int b_r, float* curSum) {
        naive_reduction_row_maxes<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output, intermediateRowMaxes, intermediatePV, b_c, b_r, curSum);
        float *l_inew = new float[b_r](); 
        for (int i = 0; i < b_r; i++) {
            float overallRowMax = fmaxf(intermediateRowMaxes[i], maxValues[i]);
            l_inew[i] = expf(maxValues[i] - overallRowMax) * sumValues[i] + expf(intermediateRowMaxes[i] - overallRowMax) * curSum[i];
        }
        naive_pv_calculation<qkv_dim>(shared_qkt, shared_v, intermediatePV, b_c, b_r);
        for (int i = 0; i < b_r; i++) {
            float overallRowMax = fmaxf(intermediateRowMaxes[i], maxValues[i]);
            for (int j = 0; j < qkv_dim; j++) {
                output[i * qkv_dim + j] *= expf(maxValues[i] - overallRowMax);
                output[i * qkv_dim + j] *= sumValues[i];
                output[i * qkv_dim + j] +=
                    (intermediatePV[i * qkv_dim + j] *
                     expf(intermediateRowMaxes[i] - overallRowMax));
                output[i * qkv_dim + j ] /= (1e-5f + l_inew[i]);
            }
            sumValues[i] = l_inew[i]; 
            maxValues[i] = overallRowMax;
        }
    }