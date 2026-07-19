#include "probs.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
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
