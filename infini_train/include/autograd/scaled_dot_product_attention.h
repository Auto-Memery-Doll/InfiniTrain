#pragma once

#include "infini_train/include/autograd/function.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace infini_train {
class Tensor;
}

namespace infini_train::autograd {
class ScaledDotProductAttention : public Function {
public:
    static constexpr char kType[] = "ScaledDotProductAttentionFunction";

    ScaledDotProductAttention(int64_t dropout_p = 0.0, bool is_causal = false, std::optional<double> scale = std::nullopt, bool enable_gqa = false) 
    : Function(kType)
    , dropout_p_(dropout_p)
    , is_causal_(is_causal)
    , scale_(scale)
    , enable_gqa_(enable_gqa) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    // flash param
    int64_t dropout_p_;
    bool is_causal_;
    float scale_;
    int num_heads_;
    int num_kv_heads_;

    // backward context
    std::vector<std::shared_ptr<Tensor>> context_;
};
}   // namespace infini_train::autograd