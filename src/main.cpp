#include <algorithm>
#include <cctype>
#include <charconv>
#include <common.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <llama.h>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ac.hpp"
#include "canon.hpp"
#include "probs.hpp"

namespace {

// constexpr const char* MODEL_PATH =
// "models/HuggingFaceTB.SmolLM3-3B-Base.Q4_K_M.gguf";
constexpr const char* MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf";

constexpr std::uint32_t PAD_SEED = 0x9E37'79B9U;

constexpr std::size_t WINDOW = 2048;
constexpr std::size_t OVERLAP = 512;

constexpr std::size_t CHUNK = 512;

constexpr std::string_view STEGO_PRIME =
    "Here is one of my favorite recipes. It is simple to make at home and "
    "always turns out delicious.\n\n";

void quiet_log(
    ggml_log_level level,
    const char* text,
    [[maybe_unused]] void* user_data
) {
    static ggml_log_level last_level = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) {
        last_level = level;
    }
    if (last_level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

llama_model* load_model() {
    llama_model_params model_params = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(MODEL_PATH, model_params);
    if (model == nullptr) {
        std::println(stderr, "error: unable to load model");
    }
    return model;
}

llama_context_params context_params(int n_ctx, int n_batch) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<std::uint32_t>(n_ctx);
    ctx_params.n_batch = static_cast<std::uint32_t>(n_batch);
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.n_threads = common_cpu_get_num_math();
    ctx_params.n_threads_batch = common_cpu_get_num_math();
    return ctx_params;
}

class WindowedEvaluator {
    llama_context* ctx;
    std::size_t n_vocab;
    llama_batch batch;
    std::vector<llama_token> history;
    std::size_t n_in_ctx = 0;

  public:
    using OnRow = std::function<void(std::vector<std::uint32_t>)>;

    WindowedEvaluator(llama_context* context, const llama_vocab* vocab)
        : ctx(context),
          n_vocab(static_cast<std::size_t>(llama_vocab_n_tokens(vocab))),
          batch(llama_batch_init(static_cast<std::int32_t>(CHUNK), 0, 1)) {}

    WindowedEvaluator(const WindowedEvaluator&) = delete;
    WindowedEvaluator& operator=(const WindowedEvaluator&) = delete;

    ~WindowedEvaluator() {
        llama_batch_free(batch);
    }

    void feed(std::span<const llama_token> tokens, const OnRow& on_row) {
        while (!tokens.empty()) {
            if (n_in_ctx == WINDOW) {
                slide();
            }
            const std::size_t take =
                std::min({tokens.size(), WINDOW - n_in_ctx, CHUNK});
            submit(tokens.first(take), on_row);
            history.insert(
                history.end(),
                tokens.begin(),
                tokens.begin() + static_cast<std::ptrdiff_t>(take)
            );
            if (history.size() > OVERLAP) {
                history.erase(
                    history.begin(),
                    history.end() - static_cast<std::ptrdiff_t>(OVERLAP)
                );
            }
            tokens = tokens.subspan(take);
        }
    }

  private:
    void slide() {
        llama_memory_clear(llama_get_memory(ctx), true);
        n_in_ctx = 0;
        std::span<const llama_token> prefix(history);
        while (!prefix.empty()) {
            const std::size_t take = std::min(prefix.size(), CHUNK);
            submit(prefix.first(take), nullptr);
            prefix = prefix.subspan(take);
        }
    }

    void submit(std::span<const llama_token> tokens, const OnRow& on_row) {
        batch.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t idx = 0; idx < tokens.size(); idx++) {
            batch.token[idx] = tokens[idx];
            batch.pos[idx] = static_cast<llama_pos>(n_in_ctx + idx);
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = 0;
            batch.logits[idx] = on_row ? 1 : 0;
        }

        if (llama_decode(ctx, batch) != 0) {
            std::println(
                stderr,
                "error: failed to decode tokens at position {}",
                n_in_ctx
            );
            std::exit(1);
        }
        n_in_ctx += tokens.size();

        if (!on_row) {
            return;
        }
        for (std::size_t idx = 0; idx < tokens.size(); idx++) {
            const float* logits =
                llama_get_logits_ith(ctx, static_cast<std::int32_t>(idx));
            if (logits == nullptr) {
                std::println(stderr, "error: failed to get logits");
                std::exit(1);
            }
            on_row(quantize_probs(std::span(logits, n_vocab)));
        }
    }
};

std::optional<std::string> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::println(stderr, "error: unable to open file '{}'", path);
        return std::nullopt;
    }

    const std::streamoff size = file.tellg();
    if (size < 0) {
        std::println(stderr, "error: unable to determine size of '{}'", path);
        return std::nullopt;
    }

    std::string content(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    file.read(content.data(), static_cast<std::streamsize>(size));
    if (file.bad()) {
        std::println(stderr, "error: failed to read file '{}'", path);
        return std::nullopt;
    }
    content.resize(static_cast<std::size_t>(file.gcount()));

    return content;
}

std::vector<llama_token> tokenize_prime(const llama_vocab* vocab) {
    if (STEGO_PRIME.empty()) {
        return {};
    }
    const int needed = -llama_tokenize(
        vocab,
        STEGO_PRIME.data(),
        static_cast<int>(STEGO_PRIME.size()),
        nullptr,
        0,
        false,
        false
    );
    if (needed <= 0) {
        return {};
    }
    std::vector<llama_token> tokens(static_cast<std::size_t>(needed));
    llama_tokenize(
        vocab,
        STEGO_PRIME.data(),
        static_cast<int>(STEGO_PRIME.size()),
        tokens.data(),
        needed,
        false,
        false
    );
    return tokens;
}

ac::Encoded run_encode(
    llama_model* model,
    std::span<const llama_token> feed,
    std::span<const llama_token> prime,
    const Perm* perm,
    bool suppress_eos,
    bool append_eos,
    bool canonical
) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const auto eos = static_cast<ac::Symbol>(llama_vocab_eos(vocab));
    const bool has_bos = add_bos && !feed.empty();
    const std::span<const llama_token> content = feed.subspan(has_bos ? 1 : 0);

    std::vector<ac::Symbol> seq;
    seq.reserve(content.size() + (append_eos ? 1 : 0));
    for (llama_token token : content) {
        seq.push_back(token_to_symbol(perm, static_cast<ac::Symbol>(token)));
    }
    if (append_eos) {
        seq.push_back(token_to_symbol(perm, eos));
    }

    std::vector<llama_token> warm;
    warm.reserve((has_bos ? 1U : 0U) + prime.size());
    if (has_bos) {
        warm.push_back(feed[0]);
    }
    warm.insert(warm.end(), prime.begin(), prime.end());

    llama_context* ctx = nullptr;
    std::optional<WindowedEvaluator> eval;
    if (!warm.empty() || !content.empty()) {
        const std::size_t n_ctx =
            std::min(WINDOW, warm.size() + content.size());
        llama_context_params ctx_params = context_params(
            static_cast<int>(n_ctx),
            static_cast<int>(std::min(CHUNK, n_ctx))
        );
        ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            std::println(stderr, "error: failed to create the context");
            std::exit(1);
        }
        eval.emplace(ctx, vocab);
    }

    std::optional<Canonical> canon;
    if (canonical) {
        canon.emplace(vocab);
    }

    std::deque<std::vector<std::uint32_t>> pending;
    if (warm.empty()) {
        pending.push_back(uniform_freqs(n_vocab));
    } else {
        std::vector<std::uint32_t> primed_row;
        eval->feed(warm, [&](std::vector<std::uint32_t> row) {
            primed_row = std::move(row);
        });
        pending.push_back(std::move(primed_row));
    }
    std::size_t fed = 0;
    std::size_t step = 0;
    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol>) {
        while (pending.empty()) {
            const std::size_t take = std::min(CHUNK, content.size() - fed);
            eval->feed(
                content.subspan(fed, take),
                [&](std::vector<std::uint32_t> row) {
                    pending.push_back(std::move(row));
                }
            );
            fed += take;
        }
        std::vector<std::uint32_t> row = std::move(pending.front());
        pending.pop_front();
        if (canon) {
            if (step > 0) {
                canon->push(static_cast<llama_token>(content[step - 1]));
            }
            mask_row(
                row,
                canonical_allow(row, *canon, static_cast<llama_token>(eos))
            );
            top_p_filter(row, eos);
        }
        step++;
        return symbol_row(std::move(row), perm, eos, suppress_eos);
    };

    ac::Encoded encoded = ac::encode(seq, prob_fn);
    if (ctx != nullptr) {
        llama_free(ctx);
    }
    return encoded;
}

struct Decoded {
    std::vector<llama_token> tokens;
    std::string text;
};

Decoded run_decode(
    llama_model* model,
    std::span<const std::uint32_t> code,
    std::size_t bits,
    std::span<const llama_token> prime,
    const Perm* perm,
    bool stego,
    bool render_special,
    bool stream
) {
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos = llama_vocab_bos(vocab);
    const auto eos = static_cast<ac::Symbol>(llama_vocab_eos(vocab));
    const ac::Symbol stop = token_to_symbol(perm, eos);

    llama_context_params ctx_params =
        context_params(static_cast<int>(WINDOW), static_cast<int>(CHUNK));
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::println(stderr, "error: failed to create the context");
        std::exit(1);
    }

    WindowedEvaluator eval(ctx, vocab);
    auto next_row = [&](llama_token token) {
        std::vector<std::uint32_t> row;
        eval.feed(std::span(&token, 1), [&](std::vector<std::uint32_t> freqs) {
            row = std::move(freqs);
        });
        return row;
    };

    std::vector<std::uint32_t> primed_row;
    bool primed = false;
    {
        std::vector<llama_token> warm;
        warm.reserve((add_bos ? 1U : 0U) + prime.size());
        if (add_bos) {
            warm.push_back(bos);
        }
        warm.insert(warm.end(), prime.begin(), prime.end());
        if (!warm.empty()) {
            eval.feed(warm, [&](std::vector<std::uint32_t> row) {
                primed_row = std::move(row);
            });
            primed = true;
        }
    }

    std::optional<Canonical> canon;
    if (stego) {
        canon.emplace(vocab);
    }

    bool drained = false;
    std::size_t pad_tokens = 0;
    std::mt19937 rng(PAD_SEED);
    ac::PadFn pad = stego ? ac::PadFn([&] {
        drained = true;
        return rng() & 1U;
    })
                          : ac::PadFn {};

    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol> seq) {
        std::vector<std::uint32_t> row;
        if (seq.empty()) {
            row = primed ? primed_row : uniform_freqs(n_vocab);
        } else {
            const auto prev =
                static_cast<llama_token>(symbol_to_token(perm, seq.back()));
            if (canon) {
                canon->push(prev);
            }
            row = next_row(prev);
        }
        if (canon) {
            mask_row(
                row,
                canonical_allow(row, *canon, static_cast<llama_token>(eos))
            );
            top_p_filter(row, eos);
        }
        std::vector<std::uint32_t> out =
            symbol_row(std::move(row), perm, eos, false);
        if (stego) {
            shape_eos(out, stop, drained, pad_tokens);
            pad_tokens += drained ? 1 : 0;
        }
        return out;
    };

    Decoded result;
    ac::OnSymbol on_symbol = [&](ac::Symbol symbol) {
        const auto token =
            static_cast<llama_token>(symbol_to_token(perm, symbol));
        result.tokens.push_back(token);
        std::string piece = common_token_to_piece(ctx, token, render_special);
        if (stream) {
            std::print("{}", piece);
            std::fflush(stdout);
        }
        result.text += piece;
    };

    ac::decode(code, bits, stop, prob_fn, on_symbol, pad);

    llama_free(ctx);
    return result;
}

int encode_mode(std::string_view text) {
    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const llama_token eos = llama_vocab_eos(vocab);
    if (eos == LLAMA_TOKEN_NULL) {
        std::println(stderr, "error: model vocab has no EOS token");
        llama_model_free(model);
        return 1;
    }

    const int n_tokens = -llama_tokenize(
        vocab,
        text.data(),
        static_cast<int>(text.size()),
        nullptr,
        0,
        true,
        false
    );

    std::vector<llama_token> tokens(static_cast<std::size_t>(n_tokens));
    if (n_tokens > 0
        && llama_tokenize(
               vocab,
               text.data(),
               static_cast<int>(text.size()),
               tokens.data(),
               static_cast<int>(tokens.size()),
               true,
               false
           ) < 0) {
        std::println(stderr, "error: unable to tokenize text");
        llama_model_free(model);
        return 1;
    }

    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const std::size_t start = (add_bos && n_tokens >= 1) ? 1 : 0;

    std::vector<ac::Symbol> seq;
    seq.reserve(tokens.size() - start + 1);
    for (std::size_t idx = start; idx < tokens.size(); idx++) {
        seq.push_back(static_cast<ac::Symbol>(tokens[idx]));
    }
    seq.push_back(static_cast<ac::Symbol>(eos));

    llama_context* ctx = nullptr;
    if (!tokens.empty()) {
        const std::size_t n_ctx = std::min(WINDOW, tokens.size());
        llama_context_params ctx_params = context_params(
            static_cast<int>(n_ctx),
            static_cast<int>(std::min(CHUNK, n_ctx))
        );
        ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            std::println(stderr, "error: failed to create the context");
            llama_model_free(model);
            return 1;
        }
    }

    std::optional<WindowedEvaluator> eval;
    if (ctx != nullptr) {
        eval.emplace(ctx, vocab);
    }

    std::deque<std::vector<std::uint32_t>> pending;
    if (start == 0) {
        pending.push_back(uniform_freqs(n_vocab));
    }
    std::size_t fed = 0;
    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol>) {
        while (pending.empty()) {
            const std::size_t take = std::min(CHUNK, tokens.size() - fed);
            eval->feed(
                std::span(tokens).subspan(fed, take),
                [&](std::vector<std::uint32_t> row) {
                    pending.push_back(std::move(row));
                }
            );
            fed += take;
        }
        std::vector<std::uint32_t> row = std::move(pending.front());
        pending.pop_front();
        return row;
    };

    ac::Encoded encoded = ac::encode(seq, prob_fn);

    const std::size_t payload_nibbles = (encoded.bits + 3) / 4;
    for (std::size_t nibble = 0; nibble < payload_nibbles; nibble++) {
        const std::uint32_t word = encoded.buffer[nibble / 8];
        const unsigned shift = 28 - (4 * (nibble % 8));
        std::print("{:x}", (word >> shift) & 0xFU);
    }
    std::println("");
    if (text.empty()) {
        std::println(stderr, "{} bits for {} tokens", encoded.bits, seq.size());
    } else {
        std::println(
            stderr,
            "{} bits for {} tokens ({} bits of text, {:.2f}% compression)",
            encoded.bits,
            seq.size(),
            text.size() * 8,
            100.0
                * (1.0
                   - static_cast<double>(encoded.bits)
                         / static_cast<double>(text.size() * 8))
        );
    }

    if (ctx != nullptr) {
        llama_free(ctx);
    }
    llama_model_free(model);

    return 0;
}

int decode_mode(std::string_view raw) {
    std::string hex;
    hex.reserve(raw.size());
    for (char ch : raw) {
        if (std::isspace(static_cast<unsigned char>(ch)) == 0) {
            hex.push_back(ch);
        }
    }

    if (hex.empty()) {
        std::println(stderr, "error: empty input");
        return 1;
    }

    const std::size_t payload_nibbles = hex.size();
    std::vector<std::uint32_t> code((payload_nibbles + 7) / 8, 0);
    for (std::size_t nibble = 0; nibble < payload_nibbles; nibble++) {
        std::uint32_t digit = 0;
        const char* first = hex.data() + nibble;
        auto [ptr, err_code] = std::from_chars(first, first + 1, digit, 16);
        if (err_code != std::errc {} || ptr != first + 1) {
            std::println(stderr, "error: invalid hex input");
            return 1;
        }
        code[nibble / 8] |= digit << (28 - (4 * (nibble % 8)));
    }

    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos = llama_vocab_bos(vocab);

    const llama_token eos = llama_vocab_eos(vocab);
    if (eos == LLAMA_TOKEN_NULL) {
        std::println(stderr, "error: model vocab has no EOS token");
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params =
        context_params(static_cast<int>(WINDOW), static_cast<int>(CHUNK));

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::println(stderr, "error: failed to create the context");
        llama_model_free(model);
        return 1;
    }

    WindowedEvaluator eval(ctx, vocab);
    auto next_row = [&](llama_token token) {
        std::vector<std::uint32_t> row;
        eval.feed(std::span(&token, 1), [&](std::vector<std::uint32_t> freqs) {
            row = std::move(freqs);
        });
        return row;
    };

    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol> seq) {
        if (seq.empty()) {
            return add_bos ? next_row(bos) : uniform_freqs(n_vocab);
        }
        return next_row(static_cast<llama_token>(seq.back()));
    };

    ac::OnSymbol on_symbol = [&](ac::Symbol symbol) {
        std::print(
            "{}",
            common_token_to_piece(ctx, static_cast<llama_token>(symbol), true)
        );
        std::fflush(stdout);
    };

    ac::decode(
        code,
        payload_nibbles * 4,
        static_cast<ac::Symbol>(eos),
        prob_fn,
        on_symbol
    );

    std::println("");

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

int stego_encode_mode(std::string_view secret) {
    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        std::println(stderr, "error: model vocab has no EOS token");
        llama_model_free(model);
        return 1;
    }
    const Perm perm =
        Perm::shuffle(static_cast<std::size_t>(llama_vocab_n_tokens(vocab)));
    const std::vector<llama_token> prime = tokenize_prime(vocab);

    int status = 1;
    std::optional<std::vector<llama_token>> secret_tokens =
        tokenize_text(vocab, secret);
    if (secret_tokens) {
        ac::Encoded code =
            run_encode(model, *secret_tokens, {}, nullptr, false, true, false);
        std::print("```\n");
        std::fflush(stdout);
        Decoded cover = run_decode(
            model,
            code.buffer,
            code.bits,
            prime,
            &perm,
            true,
            false,
            true
        );
        std::print("\n```\n");
        std::fflush(stdout);

        const bool add_bos = llama_vocab_get_add_bos(vocab);
        std::optional<std::vector<llama_token>> recheck =
            tokenize_text(vocab, cover.text);
        const std::size_t rstart =
            (recheck && add_bos && !recheck->empty()) ? 1 : 0;
        if (!recheck
            || !std::equal(
                recheck->begin() + static_cast<std::ptrdiff_t>(rstart),
                recheck->end(),
                cover.tokens.begin(),
                cover.tokens.end()
            )) {
            std::println(
                stderr,
                "error: cover is not canonically re-tokenizable"
            );
            llama_model_free(model);
            return 1;
        }

        const std::size_t start = (add_bos && !secret_tokens->empty()) ? 1 : 0;
        std::println(
            stderr,
            "hid {} secret tokens ({} bits) in {} cover tokens",
            secret_tokens->size() - start,
            code.bits,
            cover.tokens.size()
        );
        status = 0;
    }

    llama_model_free(model);
    return status;
}

int stego_decode_mode(std::string_view cover) {
    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
        std::println(stderr, "error: model vocab has no EOS token");
        llama_model_free(model);
        return 1;
    }
    const Perm perm =
        Perm::shuffle(static_cast<std::size_t>(llama_vocab_n_tokens(vocab)));
    const std::vector<llama_token> prime = tokenize_prime(vocab);

    int status = 1;
    std::optional<std::vector<llama_token>> cover_tokens =
        tokenize_text(vocab, cover);
    if (cover_tokens) {
        ac::Encoded code =
            run_encode(model, *cover_tokens, prime, &perm, true, false, true);
        run_decode(
            model,
            code.buffer,
            code.bits,
            {},
            nullptr,
            false,
            true,
            true
        );
        status = 0;
    }

    llama_model_free(model);
    return status;
}

} // namespace

int main(int argc, char** argv) {
    llama_log_set(quiet_log, nullptr);

    constexpr std::string_view usage =
        "usage: llmac <encode|decode|stego-encode|stego-decode> [-f] <input>\n"
        "  encode|decode        compress text to hex / hex back to text\n"
        "  stego-encode         hide secret text inside natural cover text\n"
        "  stego-decode         recover the secret from that cover text\n"
        "  -f, --file           read the input from the file at the given path";

    if (argc < 3 || argc > 4) {
        std::println(stderr, "{}", usage);
        return 1;
    }

    const std::string_view mode = argv[1];
    if (mode != "encode" && mode != "decode" && mode != "stego-encode"
        && mode != "stego-decode") {
        std::println(stderr, "{}", usage);
        return 1;
    }

    bool from_file = false;
    const char* input_arg = argv[2];
    if (argc == 4) {
        const std::string_view flag = argv[2];
        if (flag != "-f" && flag != "--file") {
            std::println(stderr, "{}", usage);
            return 1;
        }
        from_file = true;
        input_arg = argv[3];
    }

    std::string input;
    if (from_file) {
        std::optional<std::string> content = read_file(input_arg);
        if (!content) {
            return 1;
        }
        input = std::move(*content);
    } else {
        input = input_arg;
    }

    if (mode == "encode") {
        return encode_mode(input);
    }
    if (mode == "decode") {
        return decode_mode(input);
    }
    if (mode == "stego-encode") {
        return stego_encode_mode(input);
    }
    return stego_decode_mode(input);
}
