

//q is b_r x qkv_dim, k is b_c x qkv_dim, qkt is b_r x b_c 

#include "naive_qkt.h"


template <int qkv_dim>
void naive_qkt(float* q, float* k, float* qkt, int b_r, int b_c) {
    for (int i = 0; i<b_r;i++) {
        for (int j = 0; j < b_c; j++) {
            float dot_prod=0;//row i of q, row j of k (since tranpose of k)
            for (int d=0;d<qkv_dim;d++) {
                dot_prod += q[i*qkv_dim+d]*k[j*qkv_dim+d];
            }
            qkt[i*b_c+j] = dot_prod;
        }
    }
}