#include <iostream>
#include "fa1_forward.cuh"
#include <cuda_runtime.h>


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr<<"Usage: "<<argv[0]<<" <sequence_length>"<<std::endl;
    }
    int seq_len=std::stoi(argv[1])
    fa1_fwd_wrapper(seq_len);
    return 0;   
}