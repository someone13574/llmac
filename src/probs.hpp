#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

std::vector<std::uint32_t> quantize_probs(std::span<const float> logits);

std::vector<std::uint32_t> uniform_freqs(std::size_t count);
