#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include "fa1_forward.cuh"



template <typename T1, typename T2,int b_c, int b_r, int qkv_dim>
__device__ void calcQKT(T1* shared_q, T1* shared_k, T2* shared_qkt,int seq_len, int laneid,int warpid) {
    int req_x_tiles=ceil(b_c/TILE_Y_SIZE);
    int req_y_tiles=ceil(b_r/TILE_X_SIZE);
    int req_tiles=req_x_tiles*req_y_tiles;//# of tiles in full qk^t block output
    for (int i=warpid;i<req_tiles;i+=WARPS_PER_BLOCK) {

        int x_idx=(i)%req_x_tiles;
        int y_idx=(i)/req_x_tiles;
        int output_tile_uleft[2]={y_idx*TILE_Y_SIZE,x_idx*TILE_X_SIZE};//upper left's row, col
        T2 rC[4]={0,0,0,0};
        for (int j=0;j<qkv_dim/SHARED_Q_K_DIM;j++) {
            int q_uleft[2]={output_tile_uleft[0],j*SHARED_Q_K_DIM};
            int k_uleft[2]={output_tile_uleft[1],j*SHARED_Q_K_DIM};//storing transpose directly, row wise traversal for both Q, K tile
            T1 q_elements[8]={
                shared_q[(q_uleft[0]+laneid/4)*qkv_dim+q_uleft[1]+2*(laneid%4)],
                shared_q[(q_uleft[0]+laneid/4)*qkv_dim+q_uleft[1]+2*(laneid%4)+1],
                shared_q[(q_uleft[0]+laneid/4+8)*qkv_dim+q_uleft[1]+2*(laneid%4)],
                shared_q[(q_uleft[0]+laneid/4+8)*qkv_dim+q_uleft[1]+2*(laneid%4)+1],
                shared_q[(q_uleft[0]+laneid/4)*qkv_dim+q_uleft[1]+8+2*(laneid%4)],
                shared_q[(q_uleft[0]+laneid/4)*qkv_dim+q_uleft[1]+8+2*(laneid%4)+1],
                shared_q[(q_uleft[0]+laneid/4+8)*qkv_dim+q_uleft[1]+8+2*(laneid%4)],
                shared_q[(q_uleft[0]+laneid/4+8)*qkv_dim+q_uleft[1]+8+2*(laneid%4)+1]
            };//thank you to https://veitner.bearblog.dev/ for making the register loading a lot easier
                T1 k_elements[4]={
                    shared_k[(k_uleft[0]+laneid/4)*qkv_dim+k_uleft[1]+2*(laneid%4)],
                    shared_k[(k_uleft[0]+laneid/4)*qkv_dim+k_uleft[1]+2*(laneid%4)+1],
                    shared_k[(k_uleft[0]+laneid/4)*qkv_dim+k_uleft[1]+2*(laneid%4)+8],
                    shared_k[(k_uleft[0]+laneid/4)*qkv_dim+k_uleft[1]+2*(laneid%4)+9]//danger
                };
                //use ptx instruction!
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"//just handling the f32 accum f16 mat A,B pattern for now
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(rC[0]), "=f"(rC[1]), "=f"(rC[2]), "=f"(rC[3])
                : "r"(q_elements[0]), "r"(q_elements[1]), "r"(q_elements[2]), "r"(q_elements[3]), "r"(k_elements[0]), "r"(k_elements[1]),
                    "f"(rC[0]), "f"(rC[1]), "f"(rC[2]), "f"(rC[3]));
    }
    //store to smem
    shared_qkt[(output_tile_uleft[0]+laneid/4)*b_c+output_tile_uleft[1]+2*(laneid%4)]=rC[0];
    shared_qkt[(output_tile_uleft[0]+laneid/4)*b_c+output_tile_uleft[1]+2*(laneid%4)+1]=rC[1];
    shared_qkt[(output_tile_uleft[0]+laneid/4+8)*b_c+output_tile_uleft[1]+2*(laneid%4)]=rC[2];
    shared_qkt[(output_tile_uleft[0]+laneid/4+8)*b_c+output_tile_uleft[1]+2*(laneid%4)+1]=rC[3];
    }
}


template<typename T1, typename T2, int b_c, int b_r>
__device__ void reductionStep(T2* shared_qkt, T2* maxValues, T2* sumValues, T1* shared_v,T2* output, T2* intermediateRowMaxes, T2* intermediatePV, T1* casted_qkt, int warpid, int laneid) {
    //calculate maxValues, P_{ij} matrix, and l_ij values. split work for each row across warps

    for (int i=warpid;i<b_r;i+=WARPS_PER_BLOCK) {
        T2 m_ijProposal=-INFINITY;
        for (int j=laneid;j<b_c;j+=WARP_SIZE) {
            T2 m_ijProposal=max(m_ijProposal,shared_qkt[i][j]);
        }
        for (int offset=WARP_SIZE/2;offset>0;offset>>=1) {
            m_ijProposal=max(m_ijProposal,__shfl_down_sync(0xFFFFFFFF,m_ijProposal,offset));
        }
        if (laneid == 0) {
            maxValues[i]=max(maxValues[i],m_ijProposal);
        }
        m_ijProposal=__shfl_sync(0xFFFFFFFF,m_ijProposal,0);
        T2 runningSum=0;
        for (int j=laneid;j<b_c;j+=WARP_SIZE) {
            shared_qkt[i][j]-=m_ijProposal;
            shared_qkt[i][j]=exp(shared_qkt[i][j]);
            runningSum+=shared_qkt[i][j];
        }
        for (int offset=WARP_SIZE/2;offset>0;offset>>=1) {
            runningSum+=__shfl_down_sync(0xFFFFFFFF,runningSum,offset);//l_{ij} calculation
        }
        runningSum=__shfl_sync(0xFFFFFFFF,runningSum,0);
        T2 curMax=maxValues[i];
        T2 curRunningSum=sumValues[i];//m_i
        T2 l_inew=exp(curMax-max(curMax,m_ijProposal))*curRunningSum+exp(m_ijProposal-max(curMax,m_ijProposal))*runningSum;//l_i^{new} 
        if (laneid == 0) {
            intermediateRowMaxes[i]=m_ijProposal;
        }
        //update O_i
        for (int j=laneid;j<qkv_dim;j+=WARP_SIZE) {
            output[i][j]=(curRunningSum/l_inew)*exp(curMax-max(curMax,m_ijProposal))*output[i][j];
        }
        sumValues[i]=l_inew;
        __syncthreads();
    }
    //cast qkt to T1
    for (int i=tid;i<b_r*b_c;i+=WARP_SIZE*WARPS_PER_BLOCK) {
        casted_qkt[i/b_c][i%b_c]=shared_qkt[i/b_c][i%b_c];
    }
    __syncthreads();
    //handle p_{ij} by v_j multiplication. p_{ij} is in casted_qkt as a b_r x b_c(16x16 tiling). v_j is shared_v as a b_c x qkv_dim (16x8 tiling) 
    int req_x_tiles=ceil(qkv_dim/TILE_X_SIZE);
    int req_y_tiles=ceil(b_c/TILE_Y_SIZE);
    int req_tiles=req_x_tiles*req_y_tiles;
    for (int i=warpid;i<req_tiles;i+=WARPS_PER_BLOCK) {
        T2 rC[4]={0,0,0,0};
        int output_u_left[2]={(i)/req_x_tiles*TILE_Y_SIZE,(i)%req_x_tiles*TILE_X_SIZE};//split output tile work across warps 
        for (int j=0;j<(b_c/SHARED_Q_K_DIM);j++) {
            int p_u_left[2]={output_u_left[0],j*SHARED_Q_K_DIM};
            int v_u_left[2]={j*SHARED_Q_K_DIM,output_u_left[1]};
            T1 p_elements[8]={
                casted_qkt[(p_u_left[0]+laneid/4)*b_cp_u_left[1]+2*(laneid%4)],
                casted_qkt[(p_u_left[0]+laneid/4)*b_c+p_u_left[1]+2*(laneid%4)+1],
                casted_qkt[(p_u_left[0]+laneid/4+8)*b_c+p_u_left[1]+2*(laneid%4)],
                casted_qkt[(p_u_left[0]+laneid/4+8)*b_c+p_u_left[1]+2*(laneid%4)+1],
                casted_qkt[(p_u_left[0]+laneid/4)*b_c+p_u_left[1]+8+2*(laneid%4)],
                casted_qkt[(p_u_left[0]+laneid/4+8)*b_c+p_u_left[1]+8+2*(laneid%4)+1],
                casted_qkt[(p_u_left[0]+laneid/4)*b_c+p_u_left[1]+8+2*(laneid%4)],
                casted_qkt[(p_u_left[0]+laneid/4+8)*b_c+p_u_left[1]+8+2*(laneid%4)+1]
            };
            T1 v_elements[4]={
                shared_v[(v_u_left[0]+2*(laneid%4))*qkv_dim+v_u_left[1]+laneid/4],
                shared_v[(v_u_left[0]+2*(laneid%4)+1)*qkv_dim+v_u_left[1]+laneid/4],
                shared_v[(v_u_left[0]+2*(laneid%4)+8)*qkv_dim+v_u_left[1]+laneid/4],
                shared_v[(v_u_left[0]+2*(laneid%4)+9)*qkv_dim+v_u_left[1]+laneid/4]
            };
            //use ptx instruction!
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"//just handling the f32 accum f16 mat A,B pattern for now
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(rC[0]), "=f"(rC[1]), "=f"(rC[2]), "=f"(rC[3])
        : "r"(casted_qkt[0]), "r"(casted_qkt[1]), "r"(casted_qkt[2]), "r"(casted_qkt[3]), "r"(v_elements[0]), "r"(v_elements[1]),
            "f"(rC[0]), "f"(rC[1]), "f"(rC[2]), "f"(rC[3]));
        }
        intermediatePV[(output_u_left[0]+laneid/4)*qkv_dim+output_u_left[1]+2*(laneid%4)]=rC[0];
        intermediatePV[(output_u_left[0]+laneid/4)*qkv_dim+output_u_left[1]+2*(laneid%4)+1]=rC[1];
        intermediatePV[(output_u_left[0]+laneid/4+8)*qkv_dim+output_u_left[1]+2*(laneid%4)]=rC[2];
        intermediatePV[(output_u_left[0]+laneid/4+8)*qkv_dim+output_u_left[1]+2*(laneid%4)+1]=rC[3];
    }
    __syncthreads();
    //final O_i update
    for (int i=warpid;i<b_r;i+=WARPS_PER_BLOCK) {
        T2 coefficient=exp(intermediateRowMaxes[i]-maxValues[i])/sumValues[i];
        for (int j=laneid;j<qkv_dim;j+=WARP_SIZE) {
            output[i][j]+=coefficient*intermediatePV[i][j];
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
    extern __shared__ T2 shared_maxValues[b_r];
    extern __shared__ T2 shared_sumValues[b_r];
    extern __shared__ T2 shared_output[b_r][qkv_dim];
    extern __shared__ T2 shared_qkt[b_r][b_c];
    extern __shared__ T2 shared_intermediateRowMaxes[b_r];
    extern __shared__ T1 casted_qkt[b_r][b_c];
    extern __shared__ T2 shared_intermediatePV[b_r][qkv_dim];
    int warpid=tid/WARP_SIZE;
    int laneid=tid%WARP_SIZE;

    int head_id=bid;
    if (bid < num_heads) {//bid=head_id
        int head_prefix=head_id*seq_len*qkv_dim;
        int b_c=seq_len/(4*qkv_dim);//split k,v into tiles of this size on seq_len dim 
        int b_r=min(b_c,qkv_dim);//split q into tiles of this on seq_len dim
        int t_c=ceil(seq_len/b_c);
        int t_r=ceil(seq_len/b_r);
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
            //load in maxValues, sumValues

            __syncthreads();
            calcQKT<T1,T2,b_c,qkv_dim>(shared_q,shared_k,shared_qkt,seq_len,laneid,warpid);
            __syncthreads(); 
            //load in all required sram utils from dram 
            //first half of warps load in maxValues, second half load in sumValues
            if (warpid < WARPS_PER_BLOCK/2) {
                for(int k=tid;k<b_r;k+=(WARP_SIZE*WARPS_PER_BLOCK/2)) {
                    shared_maxValues[k]=maxValues[i*b_r+k];
                }
            } else {
                for (int k=tid-(WARP_SIZE*WARPS_PER_BLOCK/2);k<b_r;k+=(WARP_SIZE*WARPS_PER_BLOCK/2)) {
                    shared_sumValues[k]=sumValues[i*b_r+k];
            }
            }
            //collaborate on O block loading
            for (int k=tid;k<b_r*qkv_dim;k+=(WARP_SIZE*WARPS_PER_BLOCK)) {
                shared_output[k/qkv_dim][k%qkv_dim]=output[head_prefix+(b_r*i+k/qkv_dim)*qkv_dim+(k%qkv_dim)];
            }
            __syncthreads();
            reductionStep<T1,T2,b_c,qkv_dim>(shared_qkt,shared_maxValues,shared_sumValues,shared_v,shared_output,shared_intermediateRowMaxes,shared_intermediatePV,casted_qkt,warpid,laneid);
            __syncthreads();
            //write output to DRAM
            if (warpid < WARPS_PER_BLOCK/2) {
                for(int k=tid;k<b_r;k+=(WARP_SIZE*WARPS_PER_BLOCK/2)) {
                    maxValues[i*b_r+k]=shared_maxValues[k];
                }
            } else {
                for (int k=tid-(WARP_SIZE*WARPS_PER_BLOCK/2);k<b_r;k+=(WARP_SIZE*WARPS_PER_BLOCK/2)) {
                    sumValues[i*b_r+k]=shared_sumValues[k];
            }
            }
            //collaborate on O block loading
            for (int k=tid;k<b_r*qkv_dim;k+=(WARP_SIZE*WARPS_PER_BLOCK)) {
                output[head_prefix+(b_r*i+k/qkv_dim)*qkv_dim+(k%qkv_dim)]=shared_output[k/qkv_dim][k%qkv_dim];
            }
        }
    }
}


__host__ void fa1_fwd_wrapper() {

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
    auto kernel_call=fa1_fwd<__half, float, 1024, 16>;
    kernel_call <<<numBlocks, threadsPerBlock>>>(
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
}