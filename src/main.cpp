#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <llama.h>
#include <numbers>
#include <print>
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
    for (std::size_t k = 10; k > 0; --k) {
        p = std::fma(p, f, COEFFS[k - 1]);
    }

    return std::ldexp(p, static_cast<int>(xi));
}

std::vector<std::uint32_t> quantize_probs(std::span<const float> logits) {
    float max_logit = logits[0];
    for (float logit : logits) {
        max_logit = std::max(max_logit, logit);
    }

    std::vector<double> weights(logits.size());
    double sum = 0.0;
    for (std::size_t j = 0; j < logits.size(); j++) {
        double diff =
            static_cast<double>(logits[j]) - static_cast<double>(max_logit);
        weights[j] = det_exp2(diff * std::numbers::log2e);
        sum += weights[j];
    }

    auto budget = static_cast<double>(ac::QUARTER - logits.size());
    double scale = budget / sum;

    std::vector<std::uint32_t> freqs(logits.size());
    std::uint64_t total = 0;
    std::size_t top = 0;
    for (std::size_t j = 0; j < logits.size(); j++) {
        freqs[j] =
            1 + static_cast<std::uint32_t>(std::floor(weights[j] * scale));
        total += freqs[j];
        if (freqs[j] > freqs[top]) {
            top = j;
        }
    }
    freqs[top] += static_cast<std::uint32_t>(ac::QUARTER - total);

    return freqs;
}

} // namespace

int main() {
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(
        "models/HuggingFaceTB.SmolLM3-3B-Base.Q4_K_M.gguf",
        model_params
    );

    if (model == nullptr) {
        std::cerr << "error: unable to load model\n";
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const char* const text = "The quick brown fox jumped over the lazy dog.";
    const int n_tokens = -llama_tokenize(
        vocab,
        text,
        static_cast<int>(strlen(text)),
        nullptr,
        0,
        true,
        true
    );

    std::vector<llama_token> tokens(static_cast<std::size_t>(n_tokens));
    if (llama_tokenize(
            vocab,
            text,
            static_cast<int>(strlen(text)),
            tokens.data(),
            static_cast<int>(tokens.size()),
            true,
            true
        )
        < 0) {
        std::cerr << "error: unable to tokenize text\n";
        return 1;
    }

    if (n_tokens < 2) {
        std::cerr << "error: need at least two tokens to encode\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<std::uint32_t>(n_tokens);
    ctx_params.n_batch = static_cast<std::uint32_t>(n_tokens);

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::cerr << "error: failed to create the context\n";
        return 1;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int idx = 0; idx < n_tokens; ++idx) {
        batch.token[idx] = tokens[static_cast<std::size_t>(idx)];
        batch.pos[idx] = idx;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = idx + 1 < n_tokens ? 1 : 0;
    }

    int result = llama_decode(ctx, batch);
    if (result != 0) {
        std::cerr << "error/warn: failed to decode " << result << "\n";
        return 1;
    }

    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    const auto seq_len = static_cast<std::size_t>(n_tokens - 1);

    std::vector<std::uint32_t> seq_probs;
    seq_probs.reserve(seq_len * n_vocab);
    std::vector<ac::Symbol> seq(seq_len);

    for (std::size_t idx = 0; idx < seq_len; ++idx) {
        const float* logits = llama_get_logits_ith(ctx, static_cast<int>(idx));
        if (logits == nullptr) {
            std::cerr << "error: failed to get logits at position " << idx
                      << "\n";
            return 1;
        }

        std::vector<std::uint32_t> freqs =
            quantize_probs(std::span(logits, n_vocab));
        seq_probs.insert(seq_probs.end(), freqs.begin(), freqs.end());
        seq[idx] = static_cast<ac::Symbol>(tokens[idx + 1]);
    }

    ac::Encoded encoded = ac::encode(seq_probs, seq, seq_len);

    for (std::uint32_t word : encoded.buffer) {
        std::print("{:08x}", word);
    }
    std::println("");
    std::println(
        "{} bits for {} tokens ({} bits of text)",
        encoded.bits,
        seq_len,
        strlen(text) * 8
    );

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
