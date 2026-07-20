#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

std::vector<double> softmax_probs(std::span<const float> logits);

std::vector<std::uint32_t> quantize(std::span<const double> probs);

std::uint32_t quantize_step(std::size_t vocab);

std::vector<double> uniform_probs(std::size_t count);
