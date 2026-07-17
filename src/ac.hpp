#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ac {

constexpr std::uint64_t WHOLE = UINT64_C(1) << 32;
constexpr std::uint64_t HALF = WHOLE >> 1;
constexpr std::uint64_t QUARTER = WHOLE >> 2;
constexpr std::uint64_t THREE_QUARTERS = QUARTER + HALF;

struct Encoded {
    std::vector<std::uint32_t> buffer;
    std::size_t bits;
};

Encoded encode(
    std::span<std::uint32_t> seq_probs,
    std::span<std::uint32_t> seq,
    std::size_t seq_len
);

} // namespace ac
