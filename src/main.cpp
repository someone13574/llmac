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

constexpr const char* MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf";

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

bool append_model_freqs(
    llama_model* model,
    std::span<const llama_token> tokens,
    std::size_t n_vocab,
    std::vector<std::uint32_t>& seq_probs
) {
    const int n_tokens = static_cast<int>(tokens.size());
    llama_context_params ctx_params = context_params(n_tokens, n_tokens);
    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::println(stderr, "error: failed to create the context");
        return false;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    for (int idx = 0; idx < n_tokens; idx++) {
        batch.token[idx] = tokens[static_cast<std::size_t>(idx)];
        batch.pos[idx] = idx;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = idx + 1 < n_tokens ? 1 : 0;
    }

    bool ok = llama_decode(ctx, batch) == 0;
    if (!ok) {
        std::println(stderr, "error/warn: failed to decode");
    }

    for (std::size_t idx = 1; ok && idx < tokens.size(); idx++) {
        const float* logits =
            llama_get_logits_ith(ctx, static_cast<int>(idx - 1));
        if (logits == nullptr) {
            std::println(
                stderr,
                "error: failed to get logits at position {}",
                idx - 1
            );
            ok = false;
            break;
        }

        std::vector<std::uint32_t> freqs =
            quantize_probs(std::span(logits, n_vocab));
        seq_probs.insert(seq_probs.end(), freqs.begin(), freqs.end());
    }

    llama_batch_free(batch);
    llama_free(ctx);
    return ok;
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
    if (n_tokens > 0
        && llama_tokenize(
               vocab,
               text.data(),
               static_cast<int>(text.size()),
               tokens.data(),
               static_cast<int>(tokens.size()),
               true,
               true
           ) < 0) {
        std::println(stderr, "error: unable to tokenize text");
        llama_model_free(model);
        return 1;
    }

    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const std::size_t start = (add_bos && n_tokens >= 1) ? 1 : 0;
    const auto seq_len = static_cast<std::size_t>(n_tokens) - start;

    ac::Encoded encoded;
    if (seq_len > 0) {
        std::vector<std::uint32_t> seq_probs;
        seq_probs.reserve(seq_len * n_vocab);
        std::vector<ac::Symbol> seq(seq_len);
        for (std::size_t idx = 0; idx < seq_len; idx++) {
            seq[idx] = static_cast<ac::Symbol>(tokens[start + idx]);
        }

        if (start == 0) {
            std::vector<std::uint32_t> uniform = uniform_freqs(n_vocab);
            seq_probs.insert(seq_probs.end(), uniform.begin(), uniform.end());
        }

        if (tokens.size() >= 2
            && !append_model_freqs(model, tokens, n_vocab, seq_probs)) {
            llama_model_free(model);
            return 1;
        }

        encoded = ac::encode(seq_probs, seq, seq_len);
    }

    std::print("{:08x}", static_cast<std::uint32_t>(seq_len));
    const std::size_t payload_nibbles = (encoded.bits + 3) / 4;
    for (std::size_t nibble = 0; nibble < payload_nibbles; nibble++) {
        const std::uint32_t word = encoded.buffer[nibble / 8];
        const unsigned shift = 28 - (4 * (nibble % 8));
        std::print("{:x}", (word >> shift) & 0xFU);
    }
    std::println("");
    std::println(
        stderr,
        "{} bits for {} tokens ({} bits of text, {:.2f}% compression)",
        encoded.bits,
        seq_len,
        text.size() * 8,
        100.0
            * (1.0
               - static_cast<double>(encoded.bits)
                     / static_cast<double>(text.size() * 8))
    );

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

    auto parse_hex = [&](std::size_t off, std::size_t len, std::uint32_t& out) {
        const char* first = hex.data() + off;
        auto [ptr, err_code] = std::from_chars(first, first + len, out, 16);
        return err_code == std::errc {} && ptr == first + len;
    };

    constexpr std::size_t header_nibbles = 8;
    std::uint32_t seq_len_word = 0;
    if (hex.size() < header_nibbles
        || !parse_hex(0, header_nibbles, seq_len_word)) {
        std::println(stderr, "error: invalid hex input");
        return 1;
    }
    const auto seq_len = static_cast<std::size_t>(seq_len_word);

    const bool bad_size = seq_len == 0 ? hex.size() != header_nibbles
                                       : hex.size() <= header_nibbles;
    if (bad_size) {
        std::println(stderr, "error: malformed hex input");
        return 1;
    }

    const std::size_t payload_nibbles = hex.size() - header_nibbles;
    std::vector<std::uint32_t> code((payload_nibbles + 7) / 8, 0);
    for (std::size_t nibble = 0; nibble < payload_nibbles; nibble++) {
        std::uint32_t digit = 0;
        if (!parse_hex(header_nibbles + nibble, 1, digit)) {
            std::println(stderr, "error: invalid hex input");
            return 1;
        }
        code[nibble / 8] |= digit << (28 - (4 * (nibble % 8)));
    }

    if (seq_len == 0) {
        std::println("");
        return 0;
    }

    llama_model* model = load_model();
    if (model == nullptr) {
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto n_vocab = static_cast<std::size_t>(llama_vocab_n_tokens(vocab));
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos = llama_vocab_bos(vocab);

    llama_context_params ctx_params =
        context_params(static_cast<int>(seq_len) + 1, 1);

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        std::println(stderr, "error: failed to create the context");
        llama_model_free(model);
        return 1;
    }

    TokenEvaluator eval(ctx, vocab);
    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol> seq) {
        if (seq.empty()) {
            return add_bos ? eval.append(bos) : uniform_freqs(n_vocab);
        }
        return eval.append(static_cast<llama_token>(seq.back()));
    };

    ac::OnSymbol on_symbol = [&](ac::Symbol symbol) {
        std::print(
            "{}",
            common_token_to_piece(ctx, static_cast<llama_token>(symbol), true)
        );
        std::fflush(stdout);
    };

    ac::decode(code, code.size() * 32, seq_len, prob_fn, on_symbol);

    std::println("");

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
