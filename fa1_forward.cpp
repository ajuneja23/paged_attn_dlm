#include <iostream>




void fa1_fwd_wrapper() {
    // int seq_len=1024;
    // int qkv_dim=1024;
    // int num_heads=16;
    // torch::Tensor q=torch::randn({num_heads,seq_len,qkv_dim}).to(torch::kCUDA);
    // torch::Tensor k=torch::randn({num_heads,seq_len,qkv_dim}).to(torch::kCUDA);
    // torch::Tensor v=torch::randn({num_heads,seq_len,qkv_dim}).to(torch::kCUDA);
    // torch::Tensor maxValues = torch::full({num_heads, seq_len}, -std::numeric_limits<float>::infinity()).to(torch::kCUDA);
    // torch::Tensor sumValues=torch::zeros({num_heads,seq_len}).to(torch::kCUDA);
    // torch::Tensor output=torch::zeros({num_heads,seq_len,qkv_dim}).to(torch::kCUDA);
    // fa1_fwd<float16,float32,qkv_dim,num_heads>(q.data_ptr<float16>(),k.data_ptr<float16>(),v.data_ptr<float16>(),maxValues.data_ptr<float32>(),sumValues.data_ptr<float32>(),output.data_ptr<float32>(),seq_len);
    // torch::Tensor output_cpu=output.to(torch::kCPU);
    // std::cout << "output_cpu: " << output_cpu << std::endl;
}
