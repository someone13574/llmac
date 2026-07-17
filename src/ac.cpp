#include "ac.hpp"

#include <cassert>
#include <cstdint>
#include <numeric>
#include <vector>

namespace ac {

struct SymbolRange {
    std::uint32_t low;
    std::uint32_t high;
    std::uint32_t total;

    SymbolRange(std::span<std::uint32_t> probs, std::uint32_t symbol) {
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
    acc = (acc << 1) | bit;

    if (acc_size == 31) {
        acc_size = 0;
        encoded.buffer.push_back(acc);
        encoded.bits += 32;
        acc = 0;
    } else {
        acc_size += 1;
    }

    if (pending != 0) {
        pending -= 1;
        emit_bit(acc, acc_size, encoded, pending_bit, pending, pending_bit);
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
    std::span<std::uint32_t> seq,
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

        std::uint64_t range = high - low + 1;
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

} // namespace ac
