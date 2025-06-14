#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

#define TILE_X_SIZE 16
#define TILE_Y_SIZE 8//for non square tiles 
#define SQUARE_TILE_SIZE TILE_X_SIZE//for 16x16 tiles
#define SHARED_Q_K_DIM TILE_X_SIZE
/*
-TILE_X_1---TILE_X_2---TILE_X_3---TILE_X_4---TILE_X_5---TILE_X_6---TILE_X_7---TILE_X_8-
...
...
...
(row 16) TILE_X_1---TILE_X_2---TILE_X_3---TILE_X_4---TILE_X_5---TILE_X_6---TILE_X_7---TILE_X_8-

as shown this layout enables us to grab the appropriate 16x8 tile for our k^t matrix



*/ 
//using ampere m16n8k16 mma...new ports for hopper soon
#define WARPS_PER_BLOCK 4
#define WARP_SIZE 32


template <typename T1, typename T2,int b_c, int qkv_dim>
__device__ void calcQKT(T1* shared_q, T1* shared_k, int seq_len, int laneid,int warpid) {
    int req_x_tiles=ceil(b_c/TILE_X_SIZE);
    int req_y_tiles=ceil(b_r/TILE_Y_SIZE);
    int req_tiles=req_x_tiles*req_y_tiles;//output qk^t tiles 
    for (int i=0;i<req_tiles;i+=WARPS_PER_BLOCK) {
        if (i+warpid<req_tiles) {
            int x_idx=(i+warpid)%req_x_tiles;
            int y_idx=(i+warpid)/req_x_tiles;
            int output_tile_uleft[2]={y_idx*TILE_Y_SIZE,x_idx*TILE_X_SIZE};//upper left's row, col
            for (int j=0;j<qkv_dim/SHARED_Q_K_DIM;j++) {
                int q_uleft[2]={output_tile_uleft[0],output_tile_uleft[1]+j*SHARED_Q_K_DIM};
                int k_uleft[2]={output_tile_uleft[1],output_tile_uleft[0]+j*SHARED_Q_K_DIM};//LOAD IN TRANSPOSE!!! 
                //load to registers, execute ptx oh 
                int q_elements[8]=[
                    shared_q[q_uleft[0]+laneid/4][q_uleft[1]+2*(laneid%4)],
                    shared_q[q_uleft[0]+laneid/4][q_uleft[1]+2*(laneid%4)+1],
                    shared_q[q_uleft[0]+laneid/4+8][q_uleft[1]+2*(laneid%4)],
                    shared_q[q_uleft[0]+laneid/4+8][q_uleft[1]+2*(laneid%4)+1],
                    shared_q[q_uleft[0]+laneid/4][q_uleft[1]+8+2*(laneid%4)],
                    shared_q[q_uleft[0]+laneid/4][q_uleft[1]+8+2*(laneid%4)+1],
                    shared_q[q_uleft[0]+laneid/4+8][q_uleft[1]+8+2*(laneid%4)],
                    shared_q[q_uleft[0]+laneid/4+8][q_uleft[1]+8+2*(laneid%4)+1]
                    ];//thank you to https://veitner.bearblog.dev/ for making the register loading a lot easier
                    int k_elements[4]=[

                        
                    ]
        }
    }
    }

    
}
//parallelize on heads first
template<typename T1, typename T2,int qkv_dim, int num_heads>
__global__ void fa1_fwd(T1* q, T1* k, T1* v, T2* maxValues, T2* sumValues, T2* output,int seq_len)
{//q layout is (qkv_dim,seq_len,num_heads): (1, qkv_dim,qkv_dim*seq_len). same for k,v 
    int tid=threadIdx.y*blockDim.x+threadIdx.x;
    int bid=blockIdx.y*gridDim.x+blockIdx.x;
    int b_c=seq_len/(4*qkv_dim);
    int b_r=min(b_c,qkv_dim);
    extern __shared__ T1 shared_q[b_r][qkv_dim];
    extern __shared__ T1 shared_k[b_c][qkv_dim];
    extern __shared__ T1 shared_v[b_c][qkv_dim]; 
    extern __shared__ T2 shared_maxValues[b_c];
    extern __shared__ T2 shared_sumValues[b_c];
    extern __shared__ T2 shared_output[b_r][qkv_dim];
    extern __shared__ T2 shared_qkt[b_r][b_c];
    int warpid=tid/WARP_SIZE;
    int laneid=tid%WARP_SIZE;

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
                }//split k pattern
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
        }
    }
}