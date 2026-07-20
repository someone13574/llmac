#include <algorithm>
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
#include "stream.hpp"

namespace {

constexpr const char* MODEL_PATH =
    "models/HuggingFaceTB.SmolLM3-3B-Base.Q4_K_M.gguf";
// constexpr const char* MODEL_PATH = "models/Qwen3-0.6B-Q4_K_M.gguf";

constexpr std::size_t WINDOW = 2048;
constexpr std::size_t OVERLAP = 512;

constexpr std::size_t CHUNK = 512;

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
    std::uint64_t epoch = 0;

  public:
    struct Snapshot {
        std::vector<llama_token> history;
        std::size_t n_in_ctx = 0;
        std::uint64_t epoch = 0;
    };

    WindowedEvaluator(llama_context* context, const llama_vocab* vocab)
        : ctx(context),
          n_vocab(static_cast<std::size_t>(llama_vocab_n_tokens(vocab))),
          batch(llama_batch_init(static_cast<std::int32_t>(CHUNK), 0, 1)) {}

    WindowedEvaluator(const WindowedEvaluator&) = delete;
    WindowedEvaluator& operator=(const WindowedEvaluator&) = delete;

    ~WindowedEvaluator() {
        llama_batch_free(batch);
    }

    std::size_t
    submit_next(std::span<const llama_token> tokens, bool want_logits) {
        if (tokens.empty()) {
            return 0;
        }
        if (n_in_ctx == WINDOW) {
            slide();
        }
        const std::size_t take =
            std::min({tokens.size(), WINDOW - n_in_ctx, CHUNK});
        submit(tokens.first(take), want_logits);
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
        return take;
    }

    void feed(std::span<const llama_token> tokens) {
        while (!tokens.empty()) {
            tokens = tokens.subspan(submit_next(tokens, false));
        }
    }

    [[nodiscard]] std::span<const float> row(std::size_t idx) const {
        const float* logits =
            llama_get_logits_ith(ctx, static_cast<std::int32_t>(idx));
        if (logits == nullptr) {
            std::println(stderr, "error: failed to get logits");
            std::exit(1);
        }
        return {logits, n_vocab};
    }

    [[nodiscard]] Snapshot snapshot() const {
        return {.history = history, .n_in_ctx = n_in_ctx, .epoch = epoch};
    }

    void restore(const Snapshot& snap) {
        if (snap.epoch == epoch && snap.n_in_ctx <= n_in_ctx
            && llama_memory_seq_rm(
                llama_get_memory(ctx),
                0,
                static_cast<llama_pos>(snap.n_in_ctx),
                -1
            )) {
            n_in_ctx = snap.n_in_ctx;
            history = snap.history;
            return;
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        epoch += 1;
        n_in_ctx = 0;
        history.clear();
        if (!snap.history.empty()) {
            feed(snap.history);
        }
    }

  private:
    void slide() {
        llama_memory_clear(llama_get_memory(ctx), true);
        epoch += 1;
        n_in_ctx = 0;
        std::span<const llama_token> prefix(history);
        while (!prefix.empty()) {
            const std::size_t take = std::min(prefix.size(), CHUNK);
            submit(prefix.first(take), false);
            prefix = prefix.subspan(take);
        }
    }

    void submit(std::span<const llama_token> tokens, bool want_logits) {
        batch.n_tokens = static_cast<std::int32_t>(tokens.size());
        for (std::size_t idx = 0; idx < tokens.size(); idx++) {
            batch.token[idx] = tokens[idx];
            batch.pos[idx] = static_cast<llama_pos>(n_in_ctx + idx);
            batch.n_seq_id[idx] = 1;
            batch.seq_id[idx][0] = 0;
            batch.logits[idx] = want_logits ? 1 : 0;
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

    bool want_uniform = start == 0;
    std::size_t fed = 0;
    std::size_t cursor = 0;
    std::size_t rows = 0;
    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol>) {
        if (want_uniform) {
            want_uniform = false;
            return quantize(uniform_probs(n_vocab));
        }
        while (cursor == rows) {
            rows = eval->submit_next(std::span(tokens).subspan(fed), true);
            fed += rows;
            cursor = 0;
        }
        return quantize(softmax_probs(eval->row(cursor++)));
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

    std::size_t fed = 0;
    ac::GetProbs prob_fn = [&](std::span<const ac::Symbol> seq) {
        const std::size_t lead = add_bos ? 1 : 0;
        const std::size_t want = lead + seq.size();
        if (want == 0) {
            return quantize(uniform_probs(n_vocab));
        }

        std::vector<llama_token> tokens;
        tokens.reserve(want - fed);
        for (std::size_t idx = fed; idx < want; idx++) {
            tokens.push_back(
                idx < lead ? bos : static_cast<llama_token>(seq[idx - lead])
            );
        }
        fed = want;

        if (tokens.size() > 1) {
            eval.feed(std::span(tokens).first(tokens.size() - 1));
        }
        eval.submit_next(std::span(&tokens.back(), 1), true);
        return quantize(softmax_probs(eval.row(0)));
    };

    TextStream stream([&](ac::Symbol symbol) {
        return common_token_to_piece(
            ctx,
            static_cast<llama_token>(symbol),
            true
        );
    });

    WindowedEvaluator::Snapshot snapshot;
    std::size_t snapshot_fed = 0;
    ac::BlockHooks hooks {
        .save =
            [&] {
                snapshot = eval.snapshot();
                snapshot_fed = fed;
            },
        .restore =
            [&] {
                eval.restore(snapshot);
                fed = snapshot_fed;
            },
    };

    ac::Decoded decoded = ac::decode(
        code,
        payload_nibbles * 4,
        static_cast<ac::Symbol>(eos),
        quantize_step(n_vocab),
        prob_fn,
        hooks,
        stream.sink()
    );

    stream.finish();

    llama_free(ctx);
    llama_model_free(model);

    if (!decoded.ok) {
        std::println(
            stderr,
            "error: decode failed; printed output is the verified prefix"
        );
        return 1;
    }

    return 0;
}

} // namespace

int main(int argc, char** argv) {
    llama_log_set(quiet_log, nullptr);

    constexpr std::string_view usage =
        "usage: llmac <encode|decode> [-f] <input>\n"
        "  encode|decode        compress text to hex / hex back to text\n"
        "  -f, --file           read the input from the file at the given path";

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
