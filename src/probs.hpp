#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "ac.hpp"

std::vector<double> softmax_probs(std::span<const float> logits);

std::vector<std::uint32_t> quantize(std::span<const double> probs);

std::vector<double> uniform_probs(std::size_t count);

std::vector<std::uint32_t> uniform_freqs(std::size_t count);

struct Perm {
    std::vector<ac::Symbol> to_token;
    std::vector<ac::Symbol> to_symbol;

    static Perm shuffle(std::size_t n_vocab);
};

ac::Symbol token_to_symbol(const Perm* perm, ac::Symbol token);

ac::Symbol symbol_to_token(const Perm* perm, ac::Symbol symbol);

std::vector<std::uint32_t> symbol_row(
    std::vector<std::uint32_t> row,
    const Perm* perm,
    ac::Symbol eos,
    bool suppress
);

void shape_eos(
    std::vector<double>& probs,
    ac::Symbol stop,
    std::size_t committed,
    std::size_t target
);

void top_p_filter(std::vector<std::uint32_t>& probs, ac::Symbol eos);
