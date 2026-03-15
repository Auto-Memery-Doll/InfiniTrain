#include "glog/logging.h"

#include <cstddef>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <memory>
#include <vector>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
namespace infini_train::kernels::cuda {

// Template parameters for FlashAttention kernels
template <typename T, typename AccT>
struct FlashFwdParams {
    // The QKV matrices
    T *__restrict__ q_ptr;
    T *__restrict__ k_ptr;
    T *__restrict__ v_ptr;

    // The O matrix (output).
    T * __restrict__ o_ptr;

    // Intermediate results for backward pass
    AccT *__restrict__ p_ptr; // Attention weights (pre-softmax)
    AccT *__restrict__ softmax_lse_ptr; // Softmax log-sum-exp
    AccT *__restrict__ attn_weights_ptr; // Attention weights (post-softmax)

    // The stride between rows of the Q, K, V and O matrices.
    int64_t q_batch_stride;
    int64_t k_batch_stride;
    int64_t v_batch_stride;
    int64_t q_row_stride;
    int64_t k_row_stride;
    int64_t v_row_stride;
    int64_t q_head_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;
    int64_t o_batch_stride;
    int64_t o_row_stride;
    int64_t o_head_stride;
    int64_t p_batch_stride;
    int64_t p_row_stride;
    int64_t softmax_lse_batch_stride;
    int64_t attn_weights_batch_stride;
    int64_t attn_weights_row_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_softmax;

    // Dropout parameters
    float p_dropout;
    unsigned long long rng_state[2];

    // Setup context flag
    bool setup_context;
};

// Template parameters for FlashAttention backward kernels
template <typename T, typename AccT>
struct FlashBwdParams {
    // The QKV matrices
    T *__restrict__ q_ptr;
    T *__restrict__ k_ptr;
    T *__restrict__ v_ptr;

    // The dO and dQKV matrices.
    T *__restrict__ do_ptr;
    T *__restrict__ dq_ptr;
    T *__restrict__ dk_ptr;
    T *__restrict__ dv_ptr;

    // Intermediate results from forward pass
    AccT *__restrict__ p_ptr; // Attention weights (pre-softmax)
    AccT *__restrict__ softmax_lse_ptr; // Softmax log-sum-exp
    AccT *__restrict__ attn_weights_ptr; // Attention weights (post-softmax)

    // The stride between rows of the Q, K, V, dO, dQ, dK and dV matrices.
    int64_t q_batch_stride;
    int64_t k_batch_stride;
    int64_t v_batch_stride;
    int64_t q_row_stride;
    int64_t k_row_stride;
    int64_t v_row_stride;
    int64_t q_head_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;
    int64_t do_batch_stride;
    int64_t do_row_stride;
    int64_t do_head_stride;
    int64_t dq_batch_stride;
    int64_t dk_batch_stride;
    int64_t dv_batch_stride;
    int64_t dq_row_stride;
    int64_t dk_row_stride;
    int64_t dv_row_stride;
    int64_t dq_head_stride;
    int64_t dk_head_stride;
    int64_t dv_head_stride;
    int64_t p_batch_stride;
    int64_t p_row_stride;
    int64_t softmax_lse_batch_stride;
    int64_t attn_weights_batch_stride;
    int64_t attn_weights_row_stride;

    // The number of heads.
    int h, h_k;
    int h_h_k_ratio;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_rp_dropout;
    float rp_dropout;

    // Dropout parameters
    float p_dropout;
    unsigned long long rng_state[2];

    // Setup context flag
    bool setup_context;
};


// Constants
template <typename T>
__device__ __constant__ T k_zero = 0;

__device__ __constant__ float k_neg_inf = -1e10f;

// Helper function to compute causal mask
__device__ __forceinline__ bool causal_mask(int q_idx, int k_idx) {
    return q_idx >= k_idx;
}

// Template kernel for FlashAttention forward
template <typename T, typename AccT>
__global__ void FlashAttentionForwardKernel(FlashFwdParams<T, AccT> params) {
    // Get batch index and head index
    int batch_idx = blockIdx.x / params.h;
    int head_idx = blockIdx.x % params.h;
    int seq_idx = threadIdx.x;
    
    // Calculate K head index for GQA
    int k_head_idx = head_idx / params.h_h_k_ratio;
    
    // Pointers to Q, K, V for current batch and head
    T* q = params.q_ptr + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
    T* k = params.k_ptr + batch_idx * params.k_batch_stride + k_head_idx * params.k_head_stride;
    T* v = params.v_ptr + batch_idx * params.v_batch_stride + k_head_idx * params.v_head_stride;
    T* o = params.o_ptr + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride;
    
    // Pointers to intermediate results
    AccT* p = nullptr;
    AccT* softmax_lse = nullptr;
    AccT* attn_weights = nullptr;
    
    // Only initialize pointers if setup_context is true
    if (params.setup_context) {
        p = params.p_ptr + batch_idx * params.p_batch_stride + head_idx * params.h_k * params.seqlen_k;
        softmax_lse = params.softmax_lse_ptr + batch_idx * params.softmax_lse_batch_stride + head_idx;
        attn_weights = params.attn_weights_ptr + batch_idx * params.attn_weights_batch_stride + head_idx * params.attn_weights_row_stride;
    }
    
    // Process each sequence element
    if (seq_idx < params.seqlen_q) {
        // Shared memory for Q, K, V blocks
        extern __shared__ __align__(16) char smem_char[];
        T* smem = reinterpret_cast<T*>(smem_char);
        T* q_block = smem;
        T* k_block = q_block + params.d;
        T* v_block = k_block + params.d;
        
        // Load Q for current sequence element
        for (int d = 0; d < params.d; d++) {
            q_block[d] = q[seq_idx * params.q_row_stride + d];
        }
        
        // Initialize output
        AccT o_accum[128] = {0.0f};
        AccT softmax_sum = 0.0f;
        AccT max_qk = k_neg_inf;
        
        // Process each K/V sequence element to compute max QK for numerical stability
        for (int k_seq_idx = 0; k_seq_idx < params.seqlen_k; k_seq_idx++) {
            if (causal_mask(seq_idx, k_seq_idx)) {
                // Load K for current sequence element
                for (int d = 0; d < params.d; d++) {
                    k_block[d] = k[k_seq_idx * params.k_row_stride + d];
                }
                
                // Compute QK dot product
                AccT qk = 0.0f;
                for (int d = 0; d < params.d; d++) {
                    qk += static_cast<AccT>(q_block[d]) * static_cast<AccT>(k_block[d]);
                }
                
                // Apply scale
                qk *= params.scale_softmax;
                
                // Store pre-softmax attention weights
                if (params.setup_context) {
                    p[seq_idx * params.p_row_stride + k_seq_idx] = qk;
                }
                
                // Track max QK for numerical stability
                if (qk > max_qk) {
                    max_qk = qk;
                }
            } else if (params.setup_context) {
                // Store -inf for masked positions
                p[seq_idx * params.p_row_stride + k_seq_idx] = k_neg_inf;
            }
        }
        
        // Compute softmax sum with numerical stability
        for (int k_seq_idx = 0; k_seq_idx < params.seqlen_k; k_seq_idx++) {
            if (causal_mask(seq_idx, k_seq_idx)) {
                // Load K and V for current sequence element
                for (int d = 0; d < params.d; d++) {
                    k_block[d] = k[k_seq_idx * params.k_row_stride + d];
                    v_block[d] = v[k_seq_idx * params.v_row_stride + d];
                }
                
                // Compute QK dot product again (we could optimize this by storing intermediate results)
                AccT qk = 0.0f;
                for (int d = 0; d < params.d; d++) {
                    qk += static_cast<AccT>(q_block[d]) * static_cast<AccT>(k_block[d]);
                }
                
                // Apply scale and subtract max for numerical stability
                qk *= params.scale_softmax;
                AccT exp_qk = expf(qk - max_qk);
                softmax_sum += exp_qk;
                
                // Store attention weights if setup_context is true
                if (params.setup_context) {
                    attn_weights[seq_idx * params.attn_weights_row_stride + k_seq_idx] = exp_qk;
                }
                
                // Accumulate weighted V
                for (int d = 0; d < params.d; d++) {
                    o_accum[d] += exp_qk * static_cast<AccT>(v_block[d]);
                }
            } else if (params.setup_context) {
                attn_weights[seq_idx * params.attn_weights_row_stride + k_seq_idx] = 0.0f;
            }
        }
        
        // Store softmax log-sum-exp if setup_context is true
        if (params.setup_context) {
            *softmax_lse = max_qk + logf(softmax_sum);
        }
        
        // Normalize by softmax sum
        if (softmax_sum > 0.0f) {
            for (int d = 0; d < params.d; d++) {
                o[seq_idx * params.o_row_stride + d] = static_cast<T>(o_accum[d] / softmax_sum);
            }
        } else {
            for (int d = 0; d < params.d; d++) {
                o[seq_idx * params.o_row_stride + d] = static_cast<T>(0.0f);
            }
        }
    }
}

// Template kernel for FlashAttention backward
template <typename T, typename AccT>
__global__ void FlashAttentionBackwardKernel(FlashBwdParams<T, AccT> params) {
    // Get batch index and head index
    int batch_idx = blockIdx.x / params.h;
    int head_idx = blockIdx.x % params.h;
    int seq_idx = threadIdx.x;
    
    // Calculate K head index for GQA
    int k_head_idx = head_idx / params.h_h_k_ratio;
    
    // Pointers to Q, K, V, dO for current batch and head
    T* q = params.q_ptr + batch_idx * params.q_batch_stride + head_idx * params.q_head_stride;
    T* k = params.k_ptr + batch_idx * params.k_batch_stride + k_head_idx * params.k_head_stride;
    T* v = params.v_ptr + batch_idx * params.v_batch_stride + k_head_idx * params.v_head_stride;
    T* do_ptr = params.do_ptr + batch_idx * params.do_batch_stride + head_idx * params.do_head_stride;
    T* dq = params.dq_ptr + batch_idx * params.dq_batch_stride + head_idx * params.dq_head_stride;
    T* dk = params.dk_ptr + batch_idx * params.dk_batch_stride + k_head_idx * params.dk_head_stride;
    T* dv = params.dv_ptr + batch_idx * params.dv_batch_stride + k_head_idx * params.dv_head_stride;
    
    // Pointers to intermediate results from forward pass
    AccT* p = nullptr;
    AccT* softmax_lse = nullptr;
    AccT* attn_weights = nullptr;
    
    // Only initialize pointers if setup_context is true
    if (params.setup_context) {
        p = params.p_ptr + batch_idx * params.p_batch_stride + head_idx * params.h_k * params.seqlen_k;
        softmax_lse = params.softmax_lse_ptr + batch_idx * params.softmax_lse_batch_stride + head_idx;
        attn_weights = params.attn_weights_ptr + batch_idx * params.attn_weights_batch_stride + head_idx * params.attn_weights_row_stride;
    }
    
    // Process each sequence element
    if (seq_idx < params.seqlen_q) {
        // Shared memory for intermediate calculations
        extern __shared__ __align__(16) char smem_char[];
        T* smem = reinterpret_cast<T*>(smem_char);
        T* q_block = smem;
        T* k_block = q_block + params.d;
        T* v_block = k_block + params.d;
        T* do_block = v_block + params.d;
        
        // Load Q and dO for current sequence element
        for (int d = 0; d < params.d; d++) {
            q_block[d] = q[seq_idx * params.q_row_stride + d];
            do_block[d] = do_ptr[seq_idx * params.do_row_stride + d];
        }
        
        // Initialize gradients
        AccT dq_accum[128] = {0.0f};
        AccT dk_accum[128] = {0.0f};
        AccT dv_accum[128] = {0.0f};
        
        // Compute dO sum for dK and dQ gradients
        AccT do_sum = 0.0f;
        for (int d = 0; d < params.d; d++) {
            do_sum += static_cast<AccT>(do_block[d]);
        }
        
        // Get softmax sum from forward pass if available
        AccT softmax_sum = params.setup_context ? expf(*softmax_lse - k_neg_inf) : 0.0f;
        
        // Process each K/V sequence element
        for (int k_seq_idx = 0; k_seq_idx < params.seqlen_k; k_seq_idx++) {
            // Apply causal mask
            if (!causal_mask(seq_idx, k_seq_idx)) {
                continue;
            }
            
            // Load K and V for current sequence element
            for (int d = 0; d < params.d; d++) {
                k_block[d] = k[k_seq_idx * params.k_row_stride + d];
                v_block[d] = v[k_seq_idx * params.v_row_stride + d];
            }
            
            // Get attention weights from forward pass if available
            AccT softmax_val;
            if (params.setup_context) {
                // Use precomputed attention weights
                AccT exp_qk = attn_weights[seq_idx * params.attn_weights_row_stride + k_seq_idx];
                softmax_val = exp_qk / softmax_sum;
            } else {
                // Fallback to computing softmax again
                AccT qk = 0.0f;
                for (int d = 0; d < params.d; d++) {
                    qk += static_cast<AccT>(q_block[d]) * static_cast<AccT>(k_block[d]);
                }
                qk *= params.scale_softmax;
                AccT exp_qk = expf(qk);
                AccT local_softmax_sum = 0.0f;
                for (int k_idx = 0; k_idx < params.seqlen_k; k_idx++) {
                    if (causal_mask(seq_idx, k_idx)) {
                        AccT qk_val = 0.0f;
                        for (int d = 0; d < params.d; d++) {
                            qk_val += static_cast<AccT>(q_block[d]) * static_cast<AccT>(k[k_idx * params.k_row_stride + d]);
                        }
                        qk_val *= params.scale_softmax;
                        local_softmax_sum += expf(qk_val);
                    }
                }
                softmax_val = exp_qk / local_softmax_sum;
            }
            
            // Compute gradients
            for (int d = 0; d < params.d; d++) {
                // dV gradient
                dv_accum[d] += softmax_val * static_cast<AccT>(do_block[d]);
                
                // dK gradient
                dk_accum[d] += softmax_val * static_cast<AccT>(q_block[d]) * do_sum;
                
                // dQ gradient
                dq_accum[d] += softmax_val * static_cast<AccT>(k_block[d]) * do_sum;
            }
        }
        
        // Write gradients to global memory
        for (int d = 0; d < params.d; d++) {
            dq[seq_idx * params.dq_row_stride + d] = static_cast<T>(dq_accum[d] * params.scale_softmax_rp_dropout);
            dk[seq_idx * params.dk_row_stride + d] = static_cast<T>(dk_accum[d] * params.scale_softmax_rp_dropout);
            dv[seq_idx * params.dv_row_stride + d] = static_cast<T>(dv_accum[d] * params.rp_dropout);
        }
    }
}

// Template function for FlashAttention forward
template <typename T, typename AccT>
std::shared_ptr<Tensor> 
FlashAttentionForwardImpl(const std::shared_ptr<Tensor> q, 
                          const std::shared_ptr<Tensor> k, 
                          const std::shared_ptr<Tensor> v, 
                          float scale, 
                          bool causal, 
                          float dropout_p, 
                          int num_heads, 
                          int num_kv_heads, 
                          bool setup_context,
                          std::vector<std::shared_ptr<Tensor>> *context) {
    // Get device and stream
    auto device = q->GetDevice();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Get dimensions
    auto q_shape = q->Dims();
    auto k_shape = k->Dims();
    auto v_shape = v->Dims();
    
    int batch_size = q_shape[0];
    int seqlen_q = q_shape[1];
    int seqlen_k = k_shape[1];
    int d = q_shape[3];
    int h = num_heads;
    int h_k = num_kv_heads;
    int h_h_k_ratio = h / h_k;
    
    // Create output tensor
    auto o_shape = q_shape;
    auto o = std::make_shared<Tensor>(o_shape, q->Dtype(), device);
    
    // Allocate intermediate result tensors if setup_context is true
    std::shared_ptr<Tensor> p_tensor = nullptr;
    std::shared_ptr<Tensor> softmax_lse_tensor = nullptr;
    std::shared_ptr<Tensor> attn_weights_tensor = nullptr;
    
    if (setup_context) {
        // Create tensor for pre-softmax attention weights (p)
        std::vector<int64_t> p_shape = {batch_size, seqlen_q, h, seqlen_k};
        p_tensor = std::make_shared<Tensor>(p_shape, DataType::kFLOAT32, device);
        
        // Create tensor for softmax log-sum-exp
        std::vector<int64_t> softmax_lse_shape = {batch_size, h};
        softmax_lse_tensor = std::make_shared<Tensor>(softmax_lse_shape, DataType::kFLOAT32, device);
        
        // Create tensor for post-softmax attention weights
        std::vector<int64_t> attn_weights_shape = {batch_size, h, seqlen_q, seqlen_k};
        attn_weights_tensor = std::make_shared<Tensor>(attn_weights_shape, DataType::kFLOAT32, device);
    }
    
    // Prepare parameters
    FlashFwdParams<T, AccT> params;
    params.q_ptr = static_cast<T*>(q->DataPtr());
    params.k_ptr = static_cast<T*>(k->DataPtr());
    params.v_ptr = static_cast<T*>(v->DataPtr());
    params.o_ptr = static_cast<T*>(o->DataPtr());
    params.p_ptr = setup_context ? static_cast<AccT*>(p_tensor->DataPtr()) : nullptr;
    params.softmax_lse_ptr = setup_context ? static_cast<AccT*>(softmax_lse_tensor->DataPtr()) : nullptr;
    params.attn_weights_ptr = setup_context ? static_cast<AccT*>(attn_weights_tensor->DataPtr()) : nullptr;
    
    // Strides
    params.q_batch_stride = q_shape[1] * q_shape[2] * q_shape[3];
    params.k_batch_stride = k_shape[1] * k_shape[2] * k_shape[3];
    params.v_batch_stride = v_shape[1] * v_shape[2] * v_shape[3];
    params.q_row_stride = q_shape[2] * q_shape[3];
    params.k_row_stride = k_shape[2] * k_shape[3];
    params.v_row_stride = v_shape[2] * v_shape[3];
    params.q_head_stride = q_shape[3];
    params.k_head_stride = k_shape[3];
    params.v_head_stride = v_shape[3];
    params.o_batch_stride = o_shape[1] * o_shape[2] * o_shape[3];
    params.o_row_stride = o_shape[2] * o_shape[3];
    params.o_head_stride = o_shape[3];
    
    // Strides for intermediate results
    if (setup_context) {
        params.p_batch_stride = seqlen_q * h * seqlen_k;
        params.p_row_stride = h * seqlen_k;
        params.softmax_lse_batch_stride = h;
        params.attn_weights_batch_stride = h * seqlen_q * seqlen_k;
        params.attn_weights_row_stride = seqlen_q * seqlen_k;
    }
    
    // Dimensions
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h_h_k_ratio;
    
    // Scaling factors
    params.scale_softmax = scale;
    
    // Dropout parameters
    params.p_dropout = dropout_p;
    params.rng_state[0] = time(NULL);
    params.rng_state[1] = 0;
    
    // Setup context flag
    params.setup_context = setup_context;
    
    // Launch kernel
    dim3 grid(batch_size * h, 1, 1);
    dim3 block(seqlen_q, 1, 1);
    size_t smem_size = 3 * d * sizeof(T);
    FlashAttentionForwardKernel<T, AccT><<<grid, block, smem_size, stream>>>(params);
    
    // Store intermediate results in output tensor metadata if setup_context is true
    if (setup_context) {
        // We can store the intermediate tensors in the output tensor's metadata
        // This allows the backward pass to access them
        auto &con = *context;
        con[0] = p_tensor;
        con[1] = softmax_lse_tensor;
        con[2] = attn_weights_tensor;
    }
    
    return o;
}

// Template function for FlashAttention backward
template <typename T, typename AccT>
std::vector<std::shared_ptr<Tensor>> 
FlashAttentionBackwardImpl(const std::shared_ptr<Tensor> q, 
                           const std::shared_ptr<Tensor> k, 
                           const std::shared_ptr<Tensor> v, 
                           const std::shared_ptr<Tensor> o, 
                           const std::shared_ptr<Tensor> do_tensor, 
                           float scale, 
                           bool causal, 
                           float dropout_p, 
                           int num_heads, 
                           int num_kv_heads,
                           bool setup_context,
                           std::vector<std::shared_ptr<Tensor>> *context) {
    // Get device and stream
    auto device = q->GetDevice();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Get dimensions
    auto q_shape = q->Dims();
    auto k_shape = k->Dims();
    auto v_shape = v->Dims();
    auto do_shape = do_tensor->Dims();
    
    int batch_size = q_shape[0];
    int seqlen_q = q_shape[1];
    int seqlen_k = k_shape[1];
    int d = q_shape[3];
    int h = num_heads;
    int h_k = num_kv_heads;
    int h_h_k_ratio = h / h_k;
    
    // Create output tensors
    auto dq_shape = q_shape;
    auto dk_shape = k_shape;
    auto dv_shape = v_shape;
    auto dq = std::make_shared<Tensor>(dq_shape, q->Dtype(), device);
    auto dk = std::make_shared<Tensor>(dk_shape, k->Dtype(), device);
    auto dv = std::make_shared<Tensor>(dv_shape, v->Dtype(), device);
    
    // Get intermediate results from output tensor metadata if setup_context is true
    std::shared_ptr<Tensor> p_tensor = nullptr;
    std::shared_ptr<Tensor> softmax_lse_tensor = nullptr;
    std::shared_ptr<Tensor> attn_weights_tensor = nullptr;
    
    if (setup_context) {
        auto &con = *context;
        // Try to get intermediate tensors from output tensor's metadata
        p_tensor = con[0];
        softmax_lse_tensor = con[1];
        attn_weights_tensor = con[2];
        
        // If intermediate tensors are not found, fall back to not using setup_context
        if (!p_tensor || !softmax_lse_tensor || !attn_weights_tensor) {
            setup_context = false;
        }
    }
    
    // Prepare parameters
    FlashBwdParams<T, AccT> params;
    params.q_ptr = static_cast<T*>(q->DataPtr());
    params.k_ptr = static_cast<T*>(k->DataPtr());
    params.v_ptr = static_cast<T*>(v->DataPtr());
    params.do_ptr = static_cast<T*>(do_tensor->DataPtr());
    params.dq_ptr = static_cast<T*>(dq->DataPtr());
    params.dk_ptr = static_cast<T*>(dk->DataPtr());
    params.dv_ptr = static_cast<T*>(dv->DataPtr());
    params.p_ptr = setup_context ? static_cast<AccT*>(p_tensor->DataPtr()) : nullptr;
    params.softmax_lse_ptr = setup_context ? static_cast<AccT*>(softmax_lse_tensor->DataPtr()) : nullptr;
    params.attn_weights_ptr = setup_context ? static_cast<AccT*>(attn_weights_tensor->DataPtr()) : nullptr;
    
    // Strides
    params.q_batch_stride = q_shape[1] * q_shape[2] * q_shape[3];
    params.k_batch_stride = k_shape[1] * k_shape[2] * k_shape[3];
    params.v_batch_stride = v_shape[1] * v_shape[2] * v_shape[3];
    params.q_row_stride = q_shape[2] * q_shape[3];
    params.k_row_stride = k_shape[2] * k_shape[3];
    params.v_row_stride = v_shape[2] * v_shape[3];
    params.q_head_stride = q_shape[3];
    params.k_head_stride = k_shape[3];
    params.v_head_stride = v_shape[3];
    params.do_batch_stride = do_shape[1] * do_shape[2] * do_shape[3];
    params.do_row_stride = do_shape[2] * do_shape[3];
    params.do_head_stride = do_shape[3];
    params.dq_batch_stride = dq_shape[1] * dq_shape[2] * dq_shape[3];
    params.dk_batch_stride = dk_shape[1] * dk_shape[2] * dk_shape[3];
    params.dv_batch_stride = dv_shape[1] * dv_shape[2] * dv_shape[3];
    params.dq_row_stride = dq_shape[2] * dq_shape[3];
    params.dk_row_stride = dk_shape[2] * dk_shape[3];
    params.dv_row_stride = dv_shape[2] * dv_shape[3];
    params.dq_head_stride = dq_shape[3];
    params.dk_head_stride = dk_shape[3];
    params.dv_head_stride = dv_shape[3];
    
    // Strides for intermediate results
    if (setup_context) {
        params.p_batch_stride = seqlen_q * h * seqlen_k;
        params.p_row_stride = h * seqlen_k;
        params.softmax_lse_batch_stride = h;
        params.attn_weights_batch_stride = h * seqlen_q * seqlen_k;
        params.attn_weights_row_stride = seqlen_q * seqlen_k;
    }
    
    // Dimensions
    params.b = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.d = d;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h_h_k_ratio;
    
    // Scaling factors
    params.scale_softmax = scale;
    params.scale_softmax_rp_dropout = scale / (1.0f - dropout_p);
    params.rp_dropout = 1.0f / (1.0f - dropout_p);
    
    // Dropout parameters
    params.p_dropout = dropout_p;
    params.rng_state[0] = time(NULL);
    params.rng_state[1] = 0;
    
    // Setup context flag
    params.setup_context = setup_context;
    
    // Launch kernel
    dim3 grid(batch_size * h, 1, 1);
    dim3 block(seqlen_q, 1, 1);
    size_t smem_size = 4 * d * sizeof(T);
    FlashAttentionBackwardKernel<T, AccT><<<grid, block, smem_size, stream>>>(params);
    
    return {dq, dk, dv};
}

std::shared_ptr<Tensor> 
FlashAttentionForward(const std::shared_ptr<Tensor> q, 
                      const std::shared_ptr<Tensor> &k, 
                      const std::shared_ptr<Tensor> &v, 
                      float scale, 
                      bool causal, 
                      float dropout_p, 
                      int num_heads, 
                      int num_kv_heads, 
                      bool setup_context = false,
                      std::vector<std::shared_ptr<Tensor>> *context = nullptr) {
    auto dtype = q->Dtype();
    return DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
        dtype,
        [=]<typename T>() {
            using AccT = float; // Use float for accumulation to maintain precision
            return FlashAttentionForwardImpl<T, AccT>(q, k, v, scale, causal, dropout_p, num_heads, num_kv_heads, setup_context, context);
        },
        "CUDA FlashAttentionForward");
}

std::vector<std::shared_ptr<Tensor>> 
FlashAttentionBackward(const std::shared_ptr<Tensor> q, 
                       const std::shared_ptr<Tensor> k, 
                       const std::shared_ptr<Tensor> v, 
                       const std::shared_ptr<Tensor> o, 
                       const std::shared_ptr<Tensor> do_tensor, 
                       float scale, 
                       bool causal, 
                       float dropout_p, 
                       int num_heads, 
                       int num_kv_heads, 
                       bool setup_context = false,
                       std::vector<std::shared_ptr<Tensor>> *context = nullptr) {
    auto dtype = q->Dtype();
    return DispatchFunc<DataType::kFLOAT32, DataType::kBFLOAT16>(
        dtype,
        [=]<typename T>() {
            using AccT = float; // Use float for accumulation to maintain precision
            return FlashAttentionBackwardImpl<T, AccT>(q, k, v, o, do_tensor, scale, causal, dropout_p, num_heads, num_kv_heads, setup_context, context);
        },
        "CUDA FlashAttentionBackward");
}


}   // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FLASH_ATTENTION_KERNEL(kernel_name)   \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_FLASH_ATTENTION_KERNEL
