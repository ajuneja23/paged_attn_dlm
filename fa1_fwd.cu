#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>


#define TILE_SIZE 16 
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32

//parallelize on heads first
template<typename T1, typename T2,int qkv_dim, int num_heads>
__global__ void fa1_fwd(T1* q, T1* k, T1* v, T2* maxValues, T2* sumValues, T2* output,int seq_len)
{//q layout is (qkv_dim,seq_len,num_heads): (1, qkv_dim,qkv_dim*seq_len). same for k,v 
    int tid=threadIdx.y*blockDim.x+threadIdx.x;
    int bid=blockIdx.y*gridDim.x+blockIdx.x;
    int b_c=seq_len/(4*qkv_dim);
    int b_r=min(b_c,qkv_dim);
    extern __shared__ T1 shared_q[b_c][qkv_dim];
    extern __shared__ T1 shared_k[b_c][qkv_dim];
    extern __shared__ T1 shared_v[b_c][qkv_dim]; 
    extern __shared__ T2 shared_maxValues[b_c];
    extern __shared__ T2 shared_sumValues[b_c];
    extern __shared__ T2 shared_output[b_r][qkv_dim];

    int head_id=bid;
    if (bid < num_heads) {//bid=head_id
        int head_prefix=head_id*seq_len*qkv_dim;
        int b_c=seq_len/(4*qkv_dim);//split k,v into tiles of this size on seq_len dim 
        int b_r=min(b_c,qkv_dim);//split q into tiles of this on seq_len dim
        int t_c=ceil(N/b_c);
        int t_r=ceil(N/b_r);
        for (int j=0;j<t_c;j++) {//load in qkv_dim*b_c elements
            int elementsToLoad=b_c*qkv_dim;
            int seq_prefix=j*b_c*qkv_dim;
            for (int k=0;k<elementsToLoad;k+=(WARP_SIZE*WARPS_PER_BLOCK)) {
                if (k+tid<elementsToLoad) {
                    shared_k[(k+tid)/qkv_dim][(k+tid)%qkv_dim]=k[head_prefix+seq_prefix+k+tid];
                    shared_v[(k+tid)/qkv_dim][(k+tid)%qkv_dim]=v[head_prefix+seq_prefix+k+tid];
                }
            }
            __syncthreads();
            for (int i=0;i<t_r;i++) {
                int q_prefix=i*b_r*qkv_dim; 
                int elementsToLoad=b_r*qkv_dim;
                for (int k=0;k<elementsToLoad;k+=(WARP_SIZE*WARPS_PER_BLOCK)) {
                    if (k+tid<elementsToLoad) {
                        shared_q[(k+tid)/qkv_dim][(k+tid)%qkv_dim]=q[head_prefix+q_prefix+k+tid];
                    }
                }
            }
            __syncthreads();
            int cu_max=maxValues
        }
    }
}