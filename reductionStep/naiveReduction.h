#pragma once 

#include <cmath>
#include <algorithm>



template <int qkv_dim>
void naive_reduction_row_maxes(float *shared_qkt, float *maxValues,
    float *sumValues, float *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, int b_c, int b_r,
    int kElementsTracked, int qElementsTracked, float* curSum) {
        //calc m_{ij} 
        for (int i =0; i< b_r; i++) {
            intermediateRowMaxes[i] = -INFINITY;
            if (i >= qElementsTracked) { 
                continue;
            }
            for (int j = 0;j < b_c; j++) {
                if (j >= kElementsTracked) {
                    break;
                }
                intermediateRowMaxes[i]=std::max(maxValues[i],shared_qkt[i*b_c+j]);
            }
        }
        //calc s_{ij} and l_{ij}
        for (int i = 0;i<b_r;i++) {
            curSum[i] = 0.0f;
            for (int j=0;j<b_c;j++) {
                shared_qkt[i*b_c+j] -= maxValues[i]; 
                shared_qkt[i*b_c+j]=expf(shared_qkt[i*b_c+j]);
                curSum[i] += shared_qkt[i*b_c+j];
            }
        }
    }



    template <int qkv_dim>
    void naive_pv_calculation(float *shared_qkt, float *maxValues,
                              float *sumValues, float *shared_v, float *output,
                              float *intermediateRowMaxes,
                              float *intermediatePV, int b_c, int b_r,
                              int kElementsTracked, int qElementsTracked,
                              float *curSum) {
      for (int i = 0; i < b_r; i++) {
            if (i >= qElementsTracked) {
                continue;
            }
            for (int j = 0; j < qkv_dim; j++) {
                intermediatePV[i * qkv_dim + j] = 0.0f;
                for (int k = 0; k < b_c; k++) {
                    if (k >= kElementsTracked) {
                        break;
                    }
                    intermediatePV[i * qkv_dim + j] += shared_qkt[i * b_c + k] * shared_v[k * qkv_dim + j];
                }
            }
      }
}

template <int qkv_dim>
void naive_reduction(float *shared_qkt, float *maxValues,
    float *sumValues, float *shared_v, float *output,
    float *intermediateRowMaxes,
    float *intermediatePV, int b_c, int b_r,
    int kElementsTracked, int qElementsTracked, float* curSum) {
        naive_reduction_row_maxes<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output, intermediateRowMaxes, intermediatePV, b_c, b_r, kElementsTracked, qElementsTracked);
        float *l_inew=new float[b_r];
        for (int i=0;i<b_r;i++) {
            if (i>=qElementsTracked) {
                continue;
            }
            float overallRowMax=fmaxf(intermediateRowMaxes[i],maxValues[i]);
            l_inew[i]=0.0f;
            l_inew[i]+=expf(maxValues[i]-overallRowMax)*sumValues[i]+expf(intermediateRowMaxes[i]-overallRowMax)*curSum[i];
        }
        naive_pv_calculation<qkv_dim>(shared_qkt, maxValues, sumValues, shared_v, output, intermediateRowMaxes, intermediatePV, b_c, b_r, kElementsTracked, qElementsTracked, curSum);
        for (int i=0;i<b_r;i++) {
            if (i>=qElementsTracked) {
                continue;
            }
            for (int j=0;j<qkv_dim;j++) {
                float overallRowMax=fmaxf(intermediateRowMaxes[i],maxValues[i]);
                output[i*qkv_dim+j]*=expf(maxValues[i]-overallRowMax);
                output[i * qkv_dim + j] *= sumValues[i];
                output[i*qkv_dim+j]+=intermediatePV[i*qkv_dim+j]*expf(intermediateRowMaxes[i]-overallRowMax);
                sumValues[i]=l_inew[i];
                maxValues[i]=overallRowMax;
            }

            
        }
    }