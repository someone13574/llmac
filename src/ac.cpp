#include "ac.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numeric>
#include <optional>
#include <print>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace ac {

namespace {

void note(const Sink& sink, std::string_view message) {
    if (sink.note) {
        sink.note(message);
        return;
    }
    std::println(stderr, "{}", message);
}

constexpr std::size_t LEN_BITS = 16;
constexpr std::size_t FINAL_CRC_BITS = 12;
constexpr std::size_t CHECK_EVERY = 64;
constexpr std::size_t CHECKS = (BLOCK / CHECK_EVERY) - 1;
constexpr std::size_t CHECK_BITS = 4;
constexpr std::size_t MAX_SINGLES = 1536;
constexpr std::size_t PAIR_POOL = 16;
constexpr std::size_t MAX_PAIRS = 64;

std::uint32_t crc32(std::span<const Symbol> symbols) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (Symbol symbol : symbols) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            crc ^= (symbol >> shift) & 0xFFU;
            for (int bit = 0; bit < 8; bit++) {
                crc = (crc & 1U) != 0 ? (crc >> 1) ^ 0xEDB88320U : crc >> 1;
            }
        }
    }
    return ~crc;
}

class BitWriter {
    std::vector<std::uint32_t> words;
    std::size_t size = 0;

  public:
    void push(std::uint32_t bit) {
        if (size % 32 == 0) {
            words.push_back(0);
        }
        words[size / 32] |= (bit & 1U) << (31 - (size % 32));
        size += 1;
    }

    void push_bits(std::uint32_t value, std::size_t count) {
        for (std::size_t idx = 0; idx < count; idx++) {
            push((value >> (count - 1 - idx)) & 1U);
        }
    }

    void append(const BitWriter& other) {
        for (std::size_t idx = 0; idx < other.size; idx++) {
            push((other.words[idx / 32] >> (31 - (idx % 32))) & 1U);
        }
    }

    [[nodiscard]] std::size_t bits() const {
        return size;
    }

    Encoded finish() && {
        return {.buffer = std::move(words), .bits = size};
    }
};

class BitReader {
    std::span<const std::uint32_t> code;
    std::size_t bits;
    std::size_t pos = 0;

  public:
    BitReader(std::span<const std::uint32_t> code_, std::size_t bits_)
        : code(code_), bits(bits_) {}

    void seek(std::size_t pos_) {
        pos = pos_;
    }

    [[nodiscard]] std::size_t position() const {
        return pos;
    }

    std::uint32_t next() {
        if (pos >= bits) {
            pos += 1;
            return 0U;
        }

        std::uint32_t word = code[pos / 32];
        std::uint32_t bit = (word >> (31 - (pos % 32))) & 1U;
        pos += 1;
        return bit;
    }

    std::uint32_t read_bits(std::size_t count) {
        std::uint32_t value = 0;
        for (std::size_t idx = 0; idx < count; idx++) {
            value = (value << 1) | next();
        }
        return value;
    }
};

struct SymbolRange {
    Symbol symbol;
    std::uint32_t low;
    std::uint32_t high;
    std::uint32_t total;
};

SymbolRange make_range(std::span<const std::uint32_t> probs, Symbol symbol) {
    assert(symbol < probs.size());
    std::uint32_t low = std::accumulate(
        probs.begin(),
        probs.begin() + symbol,
        std::uint32_t {0}
    );
    std::uint32_t high = low + probs[symbol];
    std::uint32_t total =
        std::accumulate(probs.begin() + symbol + 1, probs.end(), high);
    assert(total <= QUARTER);
    return {.symbol = symbol, .low = low, .high = high, .total = total};
}

class BlockEncoder {
    std::uint32_t low = 0;
    std::uint32_t high = static_cast<std::uint32_t>(WHOLE - 1);
    std::uint32_t pending = 0;
    BitWriter writer;

    void emit(std::uint32_t bit) {
        writer.push(bit);
        for (; pending > 0; pending--) {
            writer.push(bit ^ 1U);
        }
    }

  public:
    void push(const SymbolRange& range) {
        std::uint64_t span = static_cast<std::uint64_t>(high) - low + 1;
        high = low + static_cast<std::uint32_t>(span * range.high / range.total)
             - 1;
        low = low + static_cast<std::uint32_t>(span * range.low / range.total);

        while (true) {
            if (high < HALF) {
                emit(0);
            } else if (low >= HALF) {
                emit(1);
                low -= HALF;
                high -= HALF;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                pending += 1;
                low -= QUARTER;
                high -= QUARTER;
            } else {
                break;
            }

            low = low << 1;
            high = (high << 1) | 1U;
        }
    }

    BitWriter finish() && {
        pending += 1;
        emit(low < QUARTER ? 0U : 1U);
        return std::move(writer);
    }
};

struct Decision {
    SymbolRange range;
    std::uint32_t margin_low;
    std::uint32_t margin_high;
};

Decision find_symbol(
    std::span<const std::uint32_t> probs,
    std::uint32_t value,
    std::uint32_t low,
    std::uint64_t range
) {
    std::uint32_t total =
        std::accumulate(probs.begin(), probs.end(), std::uint32_t {0});
    assert(total <= QUARTER);

    std::uint32_t scaled = 0;
    if (value >= low) {
        std::uint64_t offset = static_cast<std::uint64_t>(value) - low;
        std::uint64_t approx = ((offset + 1) * total - 1) / range;
        scaled =
            approx >= total ? total - 1 : static_cast<std::uint32_t>(approx);
    }

    std::uint32_t acc = 0;
    for (Symbol symbol = 0; symbol < probs.size(); symbol++) {
        std::uint32_t next = acc + probs[symbol];
        if (scaled < next) {
            return {
                .range =
                    {.symbol = symbol,
                            .low = acc,
                            .high = next,
                            .total = total},
                .margin_low = scaled - acc,
                .margin_high = next - 1 - scaled,
            };
        }
        acc = next;
    }

    assert(false && "scaled value out of range");
    return {};
}

struct CoderState {
    std::uint32_t low = 0;
    std::uint32_t high = static_cast<std::uint32_t>(WHOLE - 1);
    std::uint32_t value = 0;
};

void apply_symbol(
    CoderState& state,
    const SymbolRange& sym,
    BitReader& reader
) {
    std::uint64_t range =
        static_cast<std::uint64_t>(state.high) - state.low + 1;
    state.high = state.low
               + static_cast<std::uint32_t>(range * sym.high / sym.total) - 1;
    state.low =
        state.low + static_cast<std::uint32_t>(range * sym.low / sym.total);

    while (true) {
        if (state.low >= HALF) {
            state.low -= HALF;
            state.high -= HALF;
            state.value -= HALF;
        } else if (state.high >= HALF) {
            if (state.low < QUARTER || state.high >= THREE_QUARTERS) {
                break;
            }
            state.low -= QUARTER;
            state.high -= QUARTER;
            state.value -= QUARTER;
        }

        state.low = state.low << 1;
        state.high = (state.high << 1) | 1U;
        state.value = (state.value << 1) | reader.next();
    }
}

struct Tweak {
    std::size_t step;
    std::int32_t rel;
    std::int64_t dlow;
    std::int64_t dhigh;
};

struct StepInfo {
    Symbol symbol;
    std::uint32_t low;
    std::uint32_t high;
    std::uint32_t total;
    std::uint32_t margin_low;
    std::uint32_t margin_high;
};

struct Attempt {
    std::vector<Symbol> symbols;
    bool found_stop = false;
    std::size_t failed_check = SIZE_MAX;
};

struct BlockChecks {
    std::array<std::uint32_t, CHECKS> expect {};
    std::size_t count = 0;
};

struct FirstPass {
    std::vector<StepInfo> steps;
    std::size_t vocab = 0;
};

SymbolRange tweaked_range(
    std::span<const std::uint32_t> probs,
    const Decision& decision,
    const Tweak& tweak
) {
    SymbolRange sym = decision.range;
    if (tweak.rel != 0) {
        const auto target = static_cast<Symbol>(
            static_cast<std::int64_t>(sym.symbol) + tweak.rel
        );
        sym = make_range(probs, target);
    }

    const std::int64_t nlow = static_cast<std::int64_t>(sym.low) + tweak.dlow;
    const std::int64_t nhigh =
        static_cast<std::int64_t>(sym.high) + tweak.dhigh;
    if (nlow >= 0 && nlow < nhigh && std::cmp_less_equal(nhigh, sym.total)) {
        sym.low = static_cast<std::uint32_t>(nlow);
        sym.high = static_cast<std::uint32_t>(nhigh);
    }
    return sym;
}

std::optional<std::size_t> failed_checkpoint(
    std::size_t step,
    const BlockChecks& checks,
    std::span<const Symbol> symbols
) {
    if (step % CHECK_EVERY != CHECK_EVERY - 1
        || step / CHECK_EVERY >= checks.count) {
        return std::nullopt;
    }

    const std::size_t check = step / CHECK_EVERY;
    const std::uint32_t mask = (1U << CHECK_BITS) - 1U;
    if ((crc32(symbols) & mask) == checks.expect[check]) {
        return std::nullopt;
    }
    return check;
}

Attempt decode_block(
    BitReader& reader,
    std::size_t payload_pos,
    std::size_t bit_limit,
    bool final,
    Symbol stop,
    const BlockChecks& checks,
    const GetProbs& prob_fn,
    std::vector<Symbol>& context,
    std::span<const StepInfo> replay,
    std::span<const Tweak> tweaks,
    FirstPass* first,
    const Sink* sink
) {
    std::size_t fresh_from = 0;
    for (const Tweak& tweak : tweaks) {
        fresh_from =
            fresh_from == 0 ? tweak.step : std::min(fresh_from, tweak.step);
    }
    fresh_from = std::min(fresh_from, replay.size());

    reader.seek(payload_pos);
    CoderState state;
    state.value = reader.read_bits(32);

    Attempt attempt;
    for (std::size_t step = 0; step < BLOCK; step++) {
        if (step < fresh_from) {
            const StepInfo& info = replay[step];
            const SymbolRange sym {
                .symbol = info.symbol,
                .low = info.low,
                .high = info.high,
                .total = info.total,
            };
            attempt.symbols.push_back(sym.symbol);
            context.push_back(sym.symbol);
            apply_symbol(state, sym, reader);
            continue;
        }

        std::uint64_t range =
            static_cast<std::uint64_t>(state.high) - state.low + 1;
        std::vector<std::uint32_t> probs = prob_fn(context);
        Decision decision = find_symbol(probs, state.value, state.low, range);

        if (first != nullptr) {
            first->vocab = probs.size();
            first->steps.push_back({
                .symbol = decision.range.symbol,
                .low = decision.range.low,
                .high = decision.range.high,
                .total = decision.range.total,
                .margin_low = decision.margin_low,
                .margin_high = decision.margin_high,
            });
        }

        SymbolRange sym = decision.range;
        for (const Tweak& tweak : tweaks) {
            if (tweak.step == step) {
                sym = tweaked_range(probs, decision, tweak);
            }
        }

        attempt.symbols.push_back(sym.symbol);
        if (final && sym.symbol == stop) {
            attempt.found_stop = true;
            return attempt;
        }
        context.push_back(sym.symbol);
        if (sink != nullptr) {
            sink->emit(sym.symbol);
        }

        apply_symbol(state, sym, reader);
        if (reader.position() > bit_limit) {
            return attempt;
        }

        const std::optional<std::size_t> failed =
            failed_checkpoint(step, checks, attempt.symbols);
        if (failed) {
            attempt.failed_check = *failed;
            return attempt;
        }
    }
    return attempt;
}

void step_candidates(
    std::size_t idx,
    const StepInfo& info,
    std::uint32_t lattice,
    std::size_t vocab,
    std::vector<Tweak>& boundary,
    std::vector<Tweak>& shifts
) {
    const auto lam = static_cast<std::int64_t>(lattice);
    const std::uint32_t freq = info.high - info.low;

    if (info.symbol > 0 && info.margin_low < lattice) {
        boundary.push_back({.step = idx, .rel = -1, .dlow = 0, .dhigh = lam});
    }
    if (info.symbol + 1 < vocab && info.margin_high < lattice
        && info.high + lattice <= info.total) {
        boundary.push_back({.step = idx, .rel = 1, .dlow = -lam, .dhigh = 0});
    }

    if (info.low >= lattice) {
        shifts.push_back({.step = idx, .rel = 0, .dlow = -lam, .dhigh = 0});
    }
    if (info.margin_low >= lattice && freq > lattice) {
        shifts.push_back({.step = idx, .rel = 0, .dlow = lam, .dhigh = 0});
    }
    if (info.high + lattice <= info.total) {
        shifts.push_back({.step = idx, .rel = 0, .dlow = 0, .dhigh = lam});
    }
    if (info.margin_high >= lattice && freq > lattice) {
        shifts.push_back({.step = idx, .rel = 0, .dlow = 0, .dhigh = -lam});
    }
}

std::vector<std::vector<Tweak>> build_candidates(
    const std::vector<StepInfo>& steps,
    std::uint32_t lattice,
    std::size_t vocab,
    std::size_t window_lo
) {
    std::vector<Tweak> boundary;
    std::vector<Tweak> shifts;
    for (std::size_t rev = steps.size(); rev > 0; rev--) {
        step_candidates(
            rev - 1,
            steps[rev - 1],
            lattice,
            vocab,
            boundary,
            shifts
        );
    }

    std::vector<Tweak> singles = std::move(boundary);
    singles.insert(singles.end(), shifts.begin(), shifts.end());
    std::ranges::stable_partition(singles, [&](const Tweak& tweak) {
        return tweak.step >= window_lo;
    });
    if (singles.size() > MAX_SINGLES) {
        singles.resize(MAX_SINGLES);
    }

    std::vector<std::vector<Tweak>> candidates;
    candidates.reserve(singles.size() + MAX_PAIRS);
    for (const Tweak& tweak : singles) {
        candidates.push_back({tweak});
    }

    const std::size_t pool = std::min(singles.size(), PAIR_POOL);
    const std::size_t pair_cap =
        std::min(MAX_PAIRS, std::max<std::size_t>(8, singles.size() / 2));
    std::size_t pairs = 0;
    for (std::size_t lhs = 0; lhs < pool && pairs < pair_cap; lhs++) {
        for (std::size_t rhs = lhs + 1; rhs < pool && pairs < pair_cap; rhs++) {
            if (singles[lhs].step == singles[rhs].step) {
                continue;
            }
            candidates.push_back({singles[lhs], singles[rhs]});
            pairs += 1;
        }
    }

    return candidates;
}

struct BlockLayout {
    bool final = false;
    std::size_t payload_pos = 0;
    std::size_t check_pos = 0;
    std::size_t check_count = 0;
    std::size_t crc_pos = 0;
};

std::optional<BlockLayout>
parse_layout(BitReader& reader, std::size_t pos, std::size_t bits) {
    if (pos + 1 + FINAL_CRC_BITS > bits) {
        return std::nullopt;
    }

    reader.seek(pos);
    BlockLayout layout;
    layout.final = reader.read_bits(1) == 1;
    layout.payload_pos = pos + 1;
    if (layout.final) {
        layout.check_count = reader.read_bits(2);
        layout.payload_pos += 2;
        layout.crc_pos = bits - FINAL_CRC_BITS;
        layout.check_pos = layout.crc_pos - (layout.check_count * CHECK_BITS);
        if (layout.check_pos < layout.payload_pos) {
            return std::nullopt;
        }
        return layout;
    }

    const std::size_t len = reader.read_bits(LEN_BITS);
    layout.payload_pos += LEN_BITS;
    layout.check_pos = layout.payload_pos + len;
    layout.check_count = CHECKS;
    layout.crc_pos = layout.check_pos + (CHECKS * CHECK_BITS);
    if (layout.crc_pos + 32 > bits) {
        return std::nullopt;
    }
    return layout;
}

bool verify(
    BitReader& reader,
    const BlockLayout& layout,
    const Attempt& attempt
) {
    if (layout.final && !attempt.found_stop) {
        return false;
    }
    reader.seek(layout.crc_pos);
    if (layout.final) {
        return reader.read_bits(FINAL_CRC_BITS)
            == (crc32(attempt.symbols) & 0xFFFU);
    }
    if (attempt.symbols.size() != BLOCK) {
        return false;
    }
    return reader.read_bits(32) == crc32(attempt.symbols);
}

void emit_all(const Sink& sink, const Attempt& attempt) {
    if (!sink.emit) {
        return;
    }
    const std::size_t count =
        attempt.symbols.size() - (attempt.found_stop ? 1 : 0);
    for (std::size_t idx = 0; idx < count; idx++) {
        sink.emit(attempt.symbols[idx]);
    }
}

std::optional<Attempt> recover_block(
    BitReader& reader,
    const BlockLayout& layout,
    Symbol stop,
    const BlockChecks& checks,
    std::uint32_t lattice,
    const GetProbs& prob_fn,
    const BlockHooks& hooks,
    const Sink& sink,
    std::vector<Symbol>& context,
    std::size_t base,
    const FirstPass& first,
    std::size_t failed_check,
    std::size_t block_index
) {
    if (sink.rewind) {
        sink.rewind(base);
    }
    note(
        sink,
        std::format(
            "block {}: verification failed, searching for repair",
            block_index
        )
    );

    constexpr std::size_t WINDOW_SLACK = 8;
    std::size_t window_lo = 0;
    if (failed_check != SIZE_MAX && failed_check > 0) {
        const std::size_t window = failed_check * CHECK_EVERY;
        window_lo = window > WINDOW_SLACK ? window - WINDOW_SLACK : 0;
    }

    std::vector<std::vector<Tweak>> candidates =
        build_candidates(first.steps, lattice, first.vocab, window_lo);
    for (std::size_t idx = 0; idx < candidates.size(); idx++) {
        context.resize(base);
        if (hooks.restore) {
            hooks.restore();
        }
        Attempt retry = decode_block(
            reader,
            layout.payload_pos,
            layout.check_pos + 40,
            layout.final,
            stop,
            checks,
            prob_fn,
            context,
            first.steps,
            candidates[idx],
            nullptr,
            nullptr
        );
        if (verify(reader, layout, retry)) {
            note(
                sink,
                std::format(
                    "block {}: repaired (candidate {} of {})",
                    block_index,
                    idx + 1,
                    candidates.size()
                )
            );
            emit_all(sink, retry);
            return retry;
        }
    }

    note(
        sink,
        std::format(
            "block {}: unrepairable after {} candidates",
            block_index,
            candidates.size()
        )
    );
    return std::nullopt;
}

} // namespace

Encoded encode(std::span<const Symbol> seq, const GetProbs& prob_fn) {
    BitWriter out;

    std::size_t offset = 0;
    while (offset < seq.size()) {
        const std::size_t count = std::min(BLOCK, seq.size() - offset);
        const bool final = offset + count == seq.size();
        std::span<const Symbol> block = seq.subspan(offset, count);

        BlockEncoder coder;
        for (std::size_t idx = 0; idx < count; idx++) {
            std::vector<std::uint32_t> probs = prob_fn(seq.first(offset + idx));
            coder.push(make_range(probs, block[idx]));
        }
        BitWriter payload = std::move(coder).finish();
        assert(payload.bits() < (UINT64_C(1) << LEN_BITS));

        out.push(final ? 1U : 0U);
        if (final) {
            const std::size_t count_checks =
                count > CHECK_EVERY ? (count - 1) / CHECK_EVERY : 0;
            out.push_bits(static_cast<std::uint32_t>(count_checks), 2);
            out.append(payload);
            const std::size_t tail =
                (count_checks * CHECK_BITS) + FINAL_CRC_BITS;
            while ((out.bits() + tail) % 4 != 0) {
                out.push(0);
            }
            const std::uint32_t mask = (1U << CHECK_BITS) - 1U;
            for (std::size_t check = 0; check < count_checks; check++) {
                const std::size_t upto = (check + 1) * CHECK_EVERY;
                out.push_bits(crc32(block.first(upto)) & mask, CHECK_BITS);
            }
            out.push_bits(crc32(block) & 0xFFFU, FINAL_CRC_BITS);
        } else {
            out.push_bits(static_cast<std::uint32_t>(payload.bits()), LEN_BITS);
            out.append(payload);
            const std::uint32_t mask = (1U << CHECK_BITS) - 1U;
            for (std::size_t check = 0; check < CHECKS; check++) {
                const std::size_t upto = (check + 1) * CHECK_EVERY;
                out.push_bits(crc32(block.first(upto)) & mask, CHECK_BITS);
            }
            out.push_bits(crc32(block), 32);
        }

        offset += count;
    }

    return std::move(out).finish();
}

Decoded decode(
    std::span<const std::uint32_t> code,
    std::size_t bits,
    Symbol stop,
    std::uint32_t lattice,
    const GetProbs& prob_fn,
    const BlockHooks& hooks,
    const Sink& sink
) {
    Decoded out;
    BitReader reader(code, bits);

    std::size_t pos = 0;
    std::size_t block_index = 0;
    while (true) {
        std::optional<BlockLayout> layout = parse_layout(reader, pos, bits);
        if (!layout) {
            note(
                sink,
                std::format("error: truncated stream at block {}", block_index)
            );
            return out;
        }

        const std::size_t base = out.symbols.size();
        if (hooks.save) {
            hooks.save();
        }

        BlockChecks checks;
        checks.count = layout->check_count;
        reader.seek(layout->check_pos);
        for (std::size_t check = 0; check < checks.count; check++) {
            checks.expect[check] = reader.read_bits(CHECK_BITS);
        }

        FirstPass first;
        Attempt attempt = decode_block(
            reader,
            layout->payload_pos,
            layout->check_pos + 40,
            layout->final,
            stop,
            checks,
            prob_fn,
            out.symbols,
            {},
            {},
            &first,
            sink.emit ? &sink : nullptr
        );

        if (!verify(reader, *layout, attempt)) {
            std::optional<Attempt> repaired = recover_block(
                reader,
                *layout,
                stop,
                checks,
                lattice,
                prob_fn,
                hooks,
                sink,
                out.symbols,
                base,
                first,
                attempt.failed_check,
                block_index
            );
            if (!repaired) {
                out.symbols.resize(base);
                return out;
            }
            attempt = std::move(*repaired);
        }

        if (sink.commit) {
            sink.commit();
        }

        if (layout->final) {
            out.ok = true;
            return out;
        }
        pos = layout->crc_pos + 32;
        block_index += 1;
    }
}

} // namespace ac
