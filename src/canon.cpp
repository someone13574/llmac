#include "canon.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <llama.h>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ac.hpp"

namespace {

std::string piece_of(const llama_vocab* vocab, llama_token token) {
    std::string piece(16, '\0');
    std::int32_t written = llama_token_to_piece(
        vocab,
        token,
        piece.data(),
        static_cast<std::int32_t>(piece.size()),
        0,
        false
    );
    if (written < 0) {
        piece.resize(static_cast<std::size_t>(-written));
        written = llama_token_to_piece(
            vocab,
            token,
            piece.data(),
            static_cast<std::int32_t>(piece.size()),
            0,
            false
        );
        if (written < 0) {
            return {};
        }
    }
    piece.resize(static_cast<std::size_t>(written));
    return piece;
}

} // namespace

std::optional<std::vector<llama_token>>
tokenize_text(const llama_vocab* vocab, std::string_view text) {
    const int needed = -llama_tokenize(
        vocab,
        text.data(),
        static_cast<int>(text.size()),
        nullptr,
        0,
        true,
        false
    );
    if (needed <= 0) {
        return std::vector<llama_token> {};
    }

    std::vector<llama_token> tokens(static_cast<std::size_t>(needed));
    const int written = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int>(text.size()),
        tokens.data(),
        static_cast<int>(tokens.size()),
        true,
        false
    );
    if (written < 0) {
        std::println(stderr, "error: unable to tokenize text");
        return std::nullopt;
    }
    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}

Canonical::Canonical(const llama_vocab* vocabulary)
    : vocab(vocabulary), add_bos(llama_vocab_get_add_bos(vocabulary)) {
    const auto n_vocab =
        static_cast<std::size_t>(llama_vocab_n_tokens(vocabulary));
    pieces.resize(n_vocab);
    for (std::size_t idx = 0; idx < n_vocab; idx++) {
        pieces[idx] = piece_of(vocabulary, static_cast<llama_token>(idx));
    }
}

bool Canonical::accepts(llama_token cand) const {
    if (cand < 0 || static_cast<std::size_t>(cand) >= pieces.size()) {
        return false;
    }
    const std::string& piece = pieces[static_cast<std::size_t>(cand)];
    if (piece.empty()) {
        return false;
    }

    scratch.assign(text);
    scratch.append(piece);

    const std::optional<std::vector<llama_token>> retok =
        tokenize_text(vocab, scratch);
    if (!retok) {
        return false;
    }
    const std::size_t start = (add_bos && !retok->empty()) ? 1 : 0;
    if (retok->size() != tokens.size() + 1 + start) {
        return false;
    }
    if (retok->back() != cand) {
        return false;
    }
    return std::equal(
        tokens.begin(),
        tokens.end(),
        retok->begin() + static_cast<std::ptrdiff_t>(start)
    );
}

void Canonical::push(llama_token candidate) {
    if (candidate >= 0 && static_cast<std::size_t>(candidate) < pieces.size()) {
        text += pieces[static_cast<std::size_t>(candidate)];
    }
    tokens.push_back(candidate);
}

std::vector<std::uint8_t> canonical_allow(
    std::span<const std::uint32_t> row,
    const Canonical& canon,
    llama_token eos
) {
    const auto in_range = [&](llama_token token) {
        return token >= 0 && static_cast<std::size_t>(token) < row.size();
    };

    std::vector<std::uint32_t> order(row.size());
    std::ranges::iota(order.begin(), order.end(), 0U);
    const auto better = [&](std::uint32_t lhs, std::uint32_t rhs) {
        return row[lhs] != row[rhs] ? row[lhs] > row[rhs] : lhs < rhs;
    };

    const std::size_t window = std::min(CANDIDATES, row.size());
    std::partial_sort(
        order.begin(),
        order.begin() + static_cast<std::ptrdiff_t>(window),
        order.end(),
        better
    );

    std::vector<std::uint8_t> allow(row.size(), 0);
    if (in_range(eos)) {
        allow[static_cast<std::size_t>(eos)] = 1;
    }

    std::size_t found = 0;
    for (std::size_t idx = 0; idx < window; idx++) {
        const auto token = static_cast<llama_token>(order[idx]);
        if (token != eos && canon.accepts(token)) {
            allow[order[idx]] = 1;
            found++;
        }
    }

    if (found == 0) {
        std::ranges::sort(order.begin(), order.end(), better);
        for (std::uint32_t candidate : order) {
            const auto token = static_cast<llama_token>(candidate);
            if (token != eos && canon.accepts(token)) {
                allow[candidate] = 1;
                found++;
                break;
            }
        }
        if (found == 0) {
            for (std::uint32_t candidate : order) {
                if (static_cast<llama_token>(candidate) != eos) {
                    allow[candidate] = 1;
                    break;
                }
            }
        }
    }

    return allow;
}

void mask_row(
    std::vector<std::uint32_t>& row,
    std::span<const std::uint8_t> allow
) {
    std::uint64_t kept_mass = 0;
    std::size_t kept_count = 0;
    for (std::size_t idx = 0; idx < row.size(); idx++) {
        if (allow[idx] == 0) {
            row[idx] = 0;
        } else {
            kept_mass += row[idx];
            kept_count++;
        }
    }
    if (kept_mass == 0) {
        return;
    }

    const auto budget = static_cast<std::uint64_t>(ac::QUARTER - kept_count);
    std::uint64_t total = 0;
    std::size_t top = 0;
    for (std::size_t idx = 0; idx < row.size(); idx++) {
        if (allow[idx] == 0) {
            continue;
        }
        row[idx] = 1
                 + static_cast<std::uint32_t>(
                       static_cast<std::uint64_t>(row[idx]) * budget / kept_mass
                 );
        total += row[idx];
        if (row[idx] > row[top]) {
            top = idx;
        }
    }
    row[top] += static_cast<std::uint32_t>(ac::QUARTER - total);
}
