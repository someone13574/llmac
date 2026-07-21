#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace ac {

constexpr std::uint64_t WHOLE = UINT64_C(1) << 32;
constexpr std::uint32_t HALF = UINT32_C(1) << 31;
constexpr std::uint32_t QUARTER = HALF >> 1;
constexpr std::uint32_t THREE_QUARTERS = QUARTER + HALF;

constexpr std::size_t BLOCK = 256;

using Symbol = std::uint32_t;

struct Decoded {
    std::vector<Symbol> symbols;
    bool ok = false;
};

using GetProbs =
    std::function<std::vector<std::uint32_t>(std::span<const Symbol>)>;

struct BitSink {
    std::function<void(std::uint32_t)> push;
    std::function<void()> flush;
};

struct Sink {
    std::function<void(Symbol)> emit;
    std::function<void(std::size_t)> rewind;
    std::function<void()> commit;
    std::function<void(std::string_view)> note;
};

struct BlockHooks {
    std::function<void()> save;
    std::function<void()> restore;
};

std::size_t encode(
    std::span<const Symbol> seq,
    const GetProbs& prob_fn,
    const BitSink& sink
);

Decoded decode(
    std::span<const std::uint32_t> code,
    std::size_t bits,
    Symbol stop,
    std::uint32_t lattice,
    const GetProbs& prob_fn,
    const BlockHooks& hooks,
    const Sink& sink
);

} // namespace ac
