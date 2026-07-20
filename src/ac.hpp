#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ac {

constexpr std::uint64_t WHOLE = UINT64_C(1) << 32;
constexpr std::uint32_t HALF = UINT32_C(1) << 31;
constexpr std::uint32_t QUARTER = HALF >> 1;
constexpr std::uint32_t THREE_QUARTERS = QUARTER + HALF;

constexpr std::size_t BLOCK = 256;

using Symbol = std::uint32_t;

struct Encoded {
    std::vector<std::uint32_t> buffer;
    std::size_t bits = 0;
};

struct Decoded {
    std::vector<Symbol> symbols;
    bool ok = false;
};

using GetProbs =
    std::function<std::vector<std::uint32_t>(std::span<const Symbol>)>;

using OnSymbol = std::function<void(Symbol)>;

struct BlockHooks {
    std::function<void()> save;
    std::function<void()> restore;
};

Encoded encode(std::span<const Symbol> seq, const GetProbs& prob_fn);

Decoded decode(
    std::span<const std::uint32_t> code,
    std::size_t bits,
    Symbol stop,
    std::uint32_t lattice,
    const GetProbs& prob_fn,
    const BlockHooks& hooks,
    const OnSymbol& on_symbol
);

} // namespace ac
