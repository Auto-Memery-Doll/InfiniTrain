#include "infini_train/include/autograd/scaled_dot_product_attention.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include <memory>
#include <vector>

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];

    auto device = q->GetDevice().type();

    std::vector<std::shared_ptr<Tensor>> context(3, nullptr);    
    auto output = Dispatcher::Instance()
                    .Call<std::shared_ptr<Tensor>>(
                        {device, "FlashAttentionForward"}, 
                        q, k, v, scale_, is_causal_, dropout_p_, num_heads_, num_kv_heads_,
                        true, &context  /* setup context */
                    );

    return {output, context[0], context[1], context[2]};
}
void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors, const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];
    
    const auto &o = output_tensors[0];
    const auto &p = output_tensors[1];
    const auto &lse = output_tensors[2];
    const auto &attn_weight = output_tensors[3];

    saved_tensors_ = {q, k, v, o, p, lse, attn_weight};
}
std::vector<std::shared_ptr<Tensor>> ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    CHECK_GE(saved_tensors_.size(), 7);     // q k v o

    const auto &do_tensor = grad_outputs[0];
    const auto &q = saved_tensors_[0];
    const auto &k = saved_tensors_[1];
    const auto &v = saved_tensors_[2];
    const auto &o = saved_tensors_[3];
    const auto &p = saved_tensors_[4];
    const auto &lse = saved_tensors_[5];
    const auto &attn_weight = saved_tensors_[6];

    std::vector<std::shared_ptr<Tensor>> context = {p, lse, attn_weight};
    auto device = do_tensor->GetDevice().type();
    return Dispatcher::Instance()
            .Call<std::vector<std::shared_ptr<Tensor>>>(
                {device, "FlashAttentionBackward"}, 
                q, k, v, o, do_tensor, scale_, is_causal_, dropout_p_, num_heads_, num_kv_heads_,
                true, &context)
    ;
}

} // namespace infini_train::autograd