#include <cctype>
#include <charconv>
#include <common.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <llama.h>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ac.hpp"
#include "probs.hpp"

namespace {

constexpr const char* MODEL_PATH =
    "models/HuggingFaceTB.SmolLM3-3B-Base.Q4_K_M.gguf";

void quiet_log(ggml_log_level level, const char* text, void* /*user_data*/) {
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

llama_context_params context_params(int n_tokens, int n_batch) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<std::uint32_t>(n_tokens);
    ctx_params.n_batch = static_cast<std::uint32_t>(n_batch);
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.n_threads = common_cpu_get_num_math();
    ctx_params.n_threads_batch = common_cpu_get_num_math();
    return ctx_params;
}

class TokenEvaluator {
    llama_context* ctx;
    std::size_t n_vocab;
    llama_batch batch;
    std::size_t n_fed = 0;

  public:
    TokenEvaluator(llama_context* context, const llama_vocab* vocab)
        : ctx(context),
          n_vocab(static_cast<std::size_t>(llama_vocab_n_tokens(vocab))),
          batch(llama_batch_init(1, 0, 1)) {}

    TokenEvaluator(const TokenEvaluator&) = delete;
    TokenEvaluator& operator=(const TokenEvaluator&) = delete;

    ~TokenEvaluator() {
        llama_batch_free(batch);
    }

    std::vector<std::uint32_t> append(llama_token token) {
        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = static_cast<llama_pos>(n_fed);
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            std::println(
                stderr,
                "error: failed to decode token at position {}",
                n_fed
            );
            std::exit(1);
        }
        n_fed += 1;

        const float* logits = llama_get_logits_ith(ctx, 0);
        if (logits == nullptr) {
            std::println(stderr, "error: failed to get logits");
            std::exit(1);
        }

        return quantize_probs(std::span(logits, n_vocab));
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

int encode_mode(std::string_view text) {
    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    const int n_tokens = -llama_tokenize(
        vocab,
        text.data(),
        static_cast<int>(text.size()),
        nullptr,
        0,
        true,
        true
    );

    std::vector<llama_token> tokens(static_cast<std::size_t>(n_tokens));
    if (llama_tokenize(
            vocab,
            text.data(),
            static_cast<int>(text.size()),
            tokens.data(),
            static_cast<int>(tokens.size()),
            true,
            true
        )
        < 0) {
        std::println(stderr, "error: unable to tokenize text");
        return 1;
    }

    if (n_tokens < 2) {
        std::println(stderr, "error: need at least two tokens to encode");
        return 1;
    }

    llama_context_params ctx_params = context_params(n_tokens, n_tokens);

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::println(stderr, "error: failed to create the context");
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
        std::println(stderr, "error/warn: failed to decode {}", result);
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
            std::println(
                stderr,
                "error: failed to get logits at position {}",
                idx
            );
            return 1;
        }

        std::vector<std::uint32_t> freqs =
            quantize_probs(std::span(logits, n_vocab));
        seq_probs.insert(seq_probs.end(), freqs.begin(), freqs.end());
        seq[idx] = static_cast<ac::Symbol>(tokens[idx + 1]);
    }

    ac::Encoded encoded = ac::encode(seq_probs, seq, seq_len);

    std::print(
        "{:08x}{:08x}",
        static_cast<std::uint32_t>(seq_len),
        static_cast<std::uint32_t>(tokens[0])
    );
    for (std::uint32_t word : encoded.buffer) {
        std::print("{:08x}", word);
    }
    std::println("");
    std::println(
        stderr,
        "{} bits for {} tokens ({} bits of text)",
        encoded.bits,
        seq_len,
        text.size() * 8
    );

    llama_batch_free(batch);
    llama_free(ctx);
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

    if (hex.size() < 24 || hex.size() % 8 != 0) {
        std::println(
            stderr,
            "error: hex input must be a multiple of 8 hex digits"
        );
        return 1;
    }

    std::vector<std::uint32_t> words(hex.size() / 8);
    for (std::size_t idx = 0; idx < words.size(); idx++) {
        const char* first = hex.data() + (idx * 8);
        auto [ptr, ec] = std::from_chars(first, first + 8, words[idx], 16);
        if (ec != std::errc {} || ptr != first + 8) {
            std::println(stderr, "error: invalid hex input");
            return 1;
        }
    }

    const auto seq_len = static_cast<std::size_t>(words[0]);
    if (seq_len == 0) {
        std::println(stderr, "error: nothing to decode");
        return 1;
    }

    const auto first_token = static_cast<llama_token>(words[1]);
    std::span<const std::uint32_t> code = std::span(words).subspan(2);

    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    llama_context_params ctx_params =
        context_params(static_cast<int>(seq_len) + 1, 1);

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::println(stderr, "error: failed to create the context");
        return 1;
    }

    TokenEvaluator eval(ctx, vocab);
    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol> seq) {
        return eval.append(
            seq.empty() ? first_token : static_cast<llama_token>(seq.back())
        );
    };

    std::vector<ac::Symbol> seq =
        ac::decode(code, code.size() * 32, seq_len, prob_fn);

    std::vector<llama_token> tokens;
    tokens.reserve(seq_len + 1);
    tokens.push_back(first_token);
    for (ac::Symbol symbol : seq) {
        tokens.push_back(static_cast<llama_token>(symbol));
    }

    const std::int32_t needed = -llama_detokenize(
        vocab,
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
        nullptr,
        0,
        true,
        true
    );
    if (needed < 0) {
        std::println(stderr, "error: failed to detokenize");
        return 1;
    }

    std::string text(static_cast<std::size_t>(needed), '\0');
    const std::int32_t written = llama_detokenize(
        vocab,
        tokens.data(),
        static_cast<std::int32_t>(tokens.size()),
        text.data(),
        needed,
        true,
        true
    );
    if (written < 0) {
        std::println(stderr, "error: failed to detokenize");
        return 1;
    }
    text.resize(static_cast<std::size_t>(written));

    std::println("{}", text);

    llama_free(ctx);
    llama_model_free(model);

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    llama_log_set(quiet_log, nullptr);

    constexpr std::string_view usage =
        "usage: llmac <encode|decode> [-f] <text|hex>\n"
        "  -f, --file    read the text/hex from the file at the given path";

    if (argc < 3 || argc > 4) {
        std::println(stderr, "{}", usage);
        return 1;
    }

    const std::string_view mode = argv[1];
    if (mode != "encode" && mode != "decode") {
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
    return decode_mode(input);
}
