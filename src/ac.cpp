#include "ac.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace ac {

struct SymbolRange {
    Symbol symbol;
    std::uint32_t low;
    std::uint32_t high;
    std::uint32_t total;

    SymbolRange(
        Symbol symbol,
        std::uint32_t low,
        std::uint32_t high,
        std::uint32_t total
    )
        : symbol(symbol), low(low), high(high), total(total) {}

    SymbolRange(std::span<std::uint32_t> probs, Symbol symbol)
        : symbol(symbol) {
        assert(symbol < probs.size());
        low = std::accumulate(
            probs.begin(),
            probs.begin() + symbol,
            std::uint32_t {0}
        );
        high = low + probs[symbol];
        total = std::accumulate(probs.begin() + symbol + 1, probs.end(), high);
        assert(total <= QUARTER);
    }
};

void emit_bit(
    std::uint32_t& acc,
    std::uint32_t& acc_size,
    Encoded& encoded,
    std::uint32_t bit,
    std::uint32_t& pending,
    std::uint32_t pending_bit
) {
    while (true) {
        acc = (acc << 1) | bit;

        if (acc_size == 31) {
            acc_size = 0;
            encoded.buffer.push_back(acc);
            encoded.bits += 32;
            acc = 0;
        } else {
            acc_size += 1;
        }

        if (pending == 0) {
            break;
        }
        pending -= 1;
        bit = pending_bit;
    }
}

void finalize_encoded(
    std::uint32_t& acc,
    std::uint32_t& acc_size,
    Encoded& encoded
) {
    if (acc_size != 0) {
        acc = acc << (32 - acc_size);
        encoded.buffer.push_back(acc);
        encoded.bits += acc_size;
    }
}

Encoded encode(
    std::span<std::uint32_t> seq_probs,
    std::span<Symbol> seq,
    std::size_t seq_len
) {
    assert(seq_probs.size() % seq_len == 0);
    size_t num_symbols = seq_probs.size() / seq_len;

    std::uint32_t low = 0;
    std::uint32_t high = WHOLE - 1;

    std::uint32_t acc = 0;
    std::uint32_t acc_size = 0;
    Encoded encoded;

    uint32_t pending = 0;
    for (std::size_t idx = 0; idx < seq_len; idx++) {
        SymbolRange symbol_range = SymbolRange(
            seq_probs.subspan(idx * num_symbols, num_symbols),
            seq[idx]
        );

        std::uint64_t range = static_cast<std::uint64_t>(high) - low + 1;
        high = low
             + static_cast<std::uint32_t>(
                   range * symbol_range.high / symbol_range.total
             )
             - 1;
        low = low
            + static_cast<std::uint32_t>(
                  range * symbol_range.low / symbol_range.total
            );

        while (true) {
            if (high < HALF) {
                emit_bit(acc, acc_size, encoded, 0, pending, 1);
            } else if (low >= HALF) {
                emit_bit(acc, acc_size, encoded, 1, pending, 0);
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
            high = (high << 1) + 1;
        }
    }

    pending += 1;
    std::uint32_t bit = low < QUARTER ? 0 : 1;
    emit_bit(acc, acc_size, encoded, bit, pending, 1 - bit);

    finalize_encoded(acc, acc_size, encoded);
    return encoded;
}

namespace {

class BitReader {
    std::span<const std::uint32_t> code;
    std::size_t bits;
    std::size_t pos = 0;

  public:
    BitReader(std::span<const std::uint32_t> code, std::size_t bits)
        : code(code), bits(bits) {}

    std::uint32_t next() {
        if (pos >= bits) {
            return 0;
        }

        std::uint32_t word = code[pos / 32];
        std::uint32_t bit = (word >> (31 - (pos % 32))) & 1U;
        pos += 1;
        return bit;
    }
};

} // namespace

SymbolRange find_symbol(
    std::span<const std::uint32_t> probs,
    std::uint32_t value,
    std::uint32_t low,
    std::uint64_t range
) {
    std::uint32_t total =
        std::accumulate(probs.begin(), probs.end(), std::uint32_t {0});
    assert(total <= QUARTER);

    std::uint64_t offset = static_cast<std::uint64_t>(value) - low;
    auto scaled =
        static_cast<std::uint32_t>(((offset + 1) * total - 1) / range);

    std::uint32_t acc = 0;
    for (Symbol symbol = 0; symbol < probs.size(); symbol++) {
        std::uint32_t next = acc + probs[symbol];
        if (scaled < next) {
            return {symbol, acc, next, total};
        }
        acc = next;
    }

    assert(false && "code value out of range");
    return {0, 0, 0, 0};
}

std::vector<Symbol> decode(
    std::span<const std::uint32_t> code,
    std::size_t bits,
    std::size_t seq_len,
    const GetProbs& prob_fn
) {
    std::vector<Symbol> seq;
    seq.reserve(seq_len);

    BitReader reader(code, bits);

    std::uint32_t low = 0;
    std::uint32_t high = WHOLE - 1;

    std::uint32_t value = 0;
    for (int idx = 0; idx < 32; idx++) {
        value = (value << 1) | reader.next();
    }

    for (std::size_t idx = 0; idx < seq_len; idx++) {
        std::uint64_t range = static_cast<std::uint64_t>(high) - low + 1;
        auto probs = prob_fn(seq);

        SymbolRange symbol = find_symbol(probs, value, low, range);
        seq.push_back(symbol.symbol);

        high = low
             + static_cast<std::uint32_t>(range * symbol.high / symbol.total)
             - 1;
        low =
            low + static_cast<std::uint32_t>(range * symbol.low / symbol.total);

        while (true) {
            if (high < HALF) {
            } else if (low >= HALF) {
                low -= HALF;
                high -= HALF;
                value -= HALF;
            } else if (low >= QUARTER && high < THREE_QUARTERS) {
                low -= QUARTER;
                high -= QUARTER;
                value -= QUARTER;
            } else {
                break;
            }

            low = low << 1;
            high = (high << 1) + 1;
            value = (value << 1) | reader.next();
        }
    }

    return seq;
};

} // namespace ac
