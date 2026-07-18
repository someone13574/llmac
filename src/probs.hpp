#include <cstdint>
#include <span>
#include <vector>

std::vector<std::uint32_t> quantize_probs(std::span<const float> logits);
