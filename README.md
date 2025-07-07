# Paged Attention Support for dKV Cache Inference of Diffusion Language Models (with custom FA Implementations)

Relevant Papers: [LlaDA](https://arxiv.org/abs/2502.09992), [dKV Cache](https://arxiv.org/abs/2505.15781), [PagedAttention](https://arxiv.org/abs/2309.06180), [FlashAttention](https://arxiv.org/abs/2205.14135)

This project aims to provide an implementation of LLaDA, a modern diffusion language model with multiple modern inference optimization techniques. dKV Cache first introduced the scheme of caching the keys and values of unmasked tokens in diffusion language models. PagedAttention introduced methods to reduce memory fragmentation driven by bloated KV cache sizes. FlashAttention introduced fast fused attention kernels. With these techniques combined, I am interested to see how quick DLM inference can get.

## Progress

- [x] Basic PagedAttention Manager
- [x] FA1 Implementation
- [ ] FA2,FA3 support
- [ ] Adjust FA Implementation for Paged KV Cache
- [ ] Adjust PagedAttention Manager to maintain "unmasked K,V cache" and recompute masked token Q,K,Vs every pass while conducting global attention on all of these K,Vs in a fused kernel
