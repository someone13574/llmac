#include "probs.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "ac.hpp"

static_assert(std::numeric_limits<double>::is_iec559);
static_assert(FLT_EVAL_METHOD == 0);

namespace {

double det_exp2(double x) {
    if (x <= -32.0) {
        return 0.0;
    }

    double xi = std::floor(x);
    double f = x - xi;

    constexpr std::array<double, 11> COEFFS = {
        1.0,
        std::numbers::ln2,
        0.2402265069591007,
        0.055504108664821576,
        0.009618129107628477,
        0.0013333558146428441,
        0.00015403530393381606,
        1.5252733804059838e-05,
        1.3215486790144305e-06,
        1.0178086009239696e-07,
        7.054911620801121e-09,
    };

    double p = COEFFS[10];
    for (std::size_t idx = 10; idx > 0; idx--) {
        p = std::fma(p, f, COEFFS[idx - 1]);
    }

    return std::ldexp(p, static_cast<int>(xi));
}

constexpr std::uint32_t SHUFFLE_SEED = 0x57139A0BU;
constexpr std::size_t STEGO_RAMP = 48;
constexpr double STEGO_TOP_P = 0.95;

} // namespace

std::vector<double> uniform_probs(std::size_t count) {
    return std::vector<double>(count, 1.0);
}

std::vector<double> softmax_probs(std::span<const float> logits) {
    float max_logit = logits[0];
    for (float logit : logits) {
        max_logit = std::max(max_logit, logit);
    }

    std::vector<double> probs(logits.size());
    double sum = 0.0;
    for (std::size_t idx = 0; idx < logits.size(); idx++) {
        double diff =
            static_cast<double>(logits[idx]) - static_cast<double>(max_logit);
        probs[idx] = det_exp2(diff * std::numbers::log2e);
        sum += probs[idx];
    }

    for (double& prob : probs) {
        prob /= sum;
    }

    return probs;
}

std::vector<std::uint32_t> quantize(std::span<const double> probs) {
    double sum = 0.0;
    for (double prob : probs) {
        sum += prob;
    }

    auto budget = static_cast<double>(ac::QUARTER - probs.size());
    double scale = sum > 0.0 ? budget / sum : 0.0;

    std::vector<std::uint32_t> freqs(probs.size());
    std::uint64_t total = 0;
    std::size_t top = 0;
    for (std::size_t idx = 0; idx < probs.size(); idx++) {
        freqs[idx] =
            1 + static_cast<std::uint32_t>(std::floor(probs[idx] * scale));
        total += freqs[idx];
        if (freqs[idx] > freqs[top]) {
            top = idx;
        }
    }
    freqs[top] += static_cast<std::uint32_t>(ac::QUARTER - total);

    return freqs;
}

Perm Perm::shuffle(std::size_t n_vocab) {
    Perm perm;
    perm.to_token.resize(n_vocab);
    for (std::size_t idx = 0; idx < n_vocab; idx++) {
        perm.to_token[idx] = static_cast<ac::Symbol>(idx);
    }

    std::mt19937 rng(SHUFFLE_SEED);
    for (std::size_t idx = n_vocab; idx-- > 1;) {
        std::swap(perm.to_token[idx], perm.to_token[rng() % (idx + 1)]);
    }

    perm.to_symbol.resize(n_vocab);
    for (std::size_t idx = 0; idx < n_vocab; idx++) {
        perm.to_symbol[perm.to_token[idx]] = static_cast<ac::Symbol>(idx);
    }
    return perm;
}

ac::Symbol token_to_symbol(const Perm* perm, ac::Symbol token) {
    return perm == nullptr ? token : perm->to_symbol[token];
}

ac::Symbol symbol_to_token(const Perm* perm, ac::Symbol symbol) {
    return perm == nullptr ? symbol : perm->to_token[symbol];
}

std::vector<std::uint32_t> symbol_row(
    std::vector<std::uint32_t> row,
    const Perm* perm,
    ac::Symbol eos,
    bool suppress
) {
    if (perm != nullptr) {
        std::vector<std::uint32_t> permuted(row.size());
        for (std::size_t idx = 0; idx < row.size(); idx++) {
            permuted[idx] = row[perm->to_token[idx]];
        }
        row = std::move(permuted);
    }
    if (suppress) {
        const ac::Symbol eos_symbol = token_to_symbol(perm, eos);
        if (eos_symbol < row.size()) {
            row[eos_symbol] = 0;
        }
    }
    return row;
}

void shape_eos(
    std::vector<double>& probs,
    ac::Symbol stop,
    std::size_t committed,
    std::size_t target
) {
    if (stop >= probs.size() || committed < target) {
        return;
    }
    auto step = static_cast<double>(committed - target + 1);
    double eos_prob = std::min(1.0, step / static_cast<double>(STEGO_RAMP));

    double sum_others = 0.0;
    for (std::size_t idx = 0; idx < probs.size(); idx++) {
        if (idx != stop) {
            sum_others += probs[idx];
        }
    }
    if (sum_others > 0.0) {
        double scale = (1.0 - eos_prob) / sum_others;
        for (std::size_t idx = 0; idx < probs.size(); idx++) {
            if (idx != stop) {
                probs[idx] *= scale;
            }
        }
    }
    probs[stop] = eos_prob;
}

void top_p_filter(std::vector<std::uint32_t>& probs, ac::Symbol eos) {
    std::uint64_t total = 0;
    for (ac::Symbol symbol = 0; symbol < probs.size(); symbol++) {
        if (symbol != eos) {
            total += probs[symbol];
        }
    }
    if (total == 0) {
        return;
    }

    std::vector<ac::Symbol> order;
    for (ac::Symbol symbol = 0; symbol < probs.size(); symbol++) {
        if (symbol != eos && probs[symbol] != 0) {
            order.push_back(symbol);
        }
    }
    std::ranges::sort(order, [&](ac::Symbol lhs, ac::Symbol rhs) {
        return probs[lhs] != probs[rhs] ? probs[lhs] > probs[rhs] : lhs < rhs;
    });

    const double limit = STEGO_TOP_P * static_cast<double>(total);
    std::uint64_t cumulative = 0;
    std::size_t keep = 0;
    while (keep < order.size()) {
        cumulative += probs[order[keep]];
        keep++;
        if (static_cast<double>(cumulative) >= limit) {
            break;
        }
    }
    for (std::size_t idx = keep; idx < order.size(); idx++) {
        probs[order[idx]] = 0;
    }
}
