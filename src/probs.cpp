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

namespace {

constexpr std::uint32_t FLOOR_COUNTS = 128;
constexpr std::uint32_t GRID = 4096;

} // namespace

std::uint32_t quantize_step(std::size_t vocab) {
    const std::uint64_t floor_total =
        static_cast<std::uint64_t>(vocab) * FLOOR_COUNTS;
    return static_cast<std::uint32_t>((ac::QUARTER - floor_total) / GRID);
}

std::vector<std::uint32_t> quantize(std::span<const double> probs) {
    double sum = 0.0;
    for (double prob : probs) {
        sum += prob;
    }

    const std::uint32_t lattice = quantize_step(probs.size());
    std::vector<std::uint32_t> freqs(probs.size());
    double cum = 0.0;
    std::uint32_t prev = 0;
    for (std::size_t idx = 0; idx < probs.size(); idx++) {
        cum += sum > 0.0 ? probs[idx] : 1.0;
        const double frac =
            sum > 0.0 ? cum / sum : cum / static_cast<double>(probs.size());
        auto cell = static_cast<std::uint32_t>(std::clamp(
            std::llround(frac * GRID),
            static_cast<long long>(prev),
            static_cast<long long>(GRID)
        ));
        if (idx + 1 == probs.size()) {
            cell = GRID;
        }
        freqs[idx] = FLOOR_COUNTS + (lattice * (cell - prev));
        prev = cell;
    }

    return freqs;
}
