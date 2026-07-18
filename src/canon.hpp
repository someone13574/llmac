#pragma once

#include <cstddef>
#include <cstdint>
#include <llama.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

inline constexpr std::size_t CANDIDATES = 128;

std::optional<std::vector<llama_token>>
tokenize_text(const llama_vocab* vocab, std::string_view text);

class Canonical {
    const llama_vocab* vocab;
    bool add_bos;
    std::vector<std::string> pieces;
    std::string text;
    std::vector<llama_token> tokens;
    mutable std::string scratch;

  public:
    explicit Canonical(const llama_vocab* vocabulary);

    bool accepts(llama_token cand) const;
    void push(llama_token cand);

    const std::string& cover_text() const {
        return text;
    }

    const std::vector<llama_token>& cover_tokens() const {
        return tokens;
    }
};

std::vector<std::uint8_t> canonical_allow(
    std::span<const std::uint32_t> row,
    const Canonical& canon,
    llama_token eos
);

void mask_row(
    std::vector<std::uint32_t>& row,
    std::span<const std::uint8_t> allow
);
