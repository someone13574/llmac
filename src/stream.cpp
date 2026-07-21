#include "stream.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <print>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>

#include "ac.hpp"

namespace {

constexpr std::size_t DEFAULT_WIDTH = 80;
constexpr std::size_t TAB_STOP = 8;

bool is_continuation(char byte) {
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

std::size_t terminal_width() {
    winsize size {};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 1) {
        return size.ws_col;
    }
    return DEFAULT_WIDTH;
}

} // namespace

ac::BitSink HexStream::sink() {
    return {
        .push = [this](std::uint32_t bit) { push(bit); },
        .flush = [] { std::fflush(stdout); },
    };
}

void HexStream::finish() {
    while (held != 0) {
        push(0);
    }
    std::println("");
    std::fflush(stdout);
}

void HexStream::push(std::uint32_t bit) {
    carry = (carry << 1) | (bit & 1U);
    held += 1;
    if (held == 4) {
        std::print("{:x}", carry);
        carry = 0;
        held = 0;
    }
}

TextStream::TextStream(Render render_)
    : render(std::move(render_)), interactive(isatty(STDOUT_FILENO) != 0),
      width(terminal_width()) {}

ac::Sink TextStream::sink() {
    return {
        .emit = [this](ac::Symbol symbol) { emit(symbol); },
        .rewind = [this](std::size_t count) { rewind(count); },
        .commit = [this] { commit(); },
        .note = [this](std::string_view message) { note(message); },
    };
}

void TextStream::finish() {
    if (!interactive) {
        for (const std::string& piece : pieces) {
            std::print("{}", piece);
        }
        pieces.clear();
    }
    std::println("");
    std::fflush(stdout);

    for (const std::string& message : notes) {
        std::println(stderr, "{}", message);
    }
    notes.clear();
}

void TextStream::emit(ac::Symbol symbol) {
    std::string piece = render(symbol);
    if (interactive) {
        write(piece);
        std::fflush(stdout);
    }
    pieces.push_back(std::move(piece));
}

void TextStream::rewind(std::size_t count) {
    const std::size_t keep =
        count > committed ? std::min(count - committed, pieces.size()) : 0;
    if (keep == pieces.size()) {
        return;
    }

    if (interactive) {
        if (rows > 0) {
            std::print("\x1b[{}A", rows);
        }
        std::print("\r\x1b[J");

        const std::string prefix = start_row;
        row.clear();
        col = 0;
        rows = 0;
        write(prefix);
        rows = 0;

        for (std::size_t idx = 0; idx < keep; idx++) {
            write(pieces[idx]);
        }
        std::fflush(stdout);
    }

    pieces.resize(keep);
}

void TextStream::commit() {
    if (!interactive) {
        for (const std::string& piece : pieces) {
            std::print("{}", piece);
        }
        std::fflush(stdout);
    }

    committed += pieces.size();
    pieces.clear();
    start_row = row;
    rows = 0;
}

void TextStream::note(std::string_view message) {
    if (interactive) {
        notes.emplace_back(message);
        return;
    }
    std::println(stderr, "{}", message);
}

void TextStream::write(std::string_view text) {
    advance(text);
    std::print("{}", text);
}

void TextStream::advance(std::string_view text) {
    for (char byte : text) {
        if (byte == '\n') {
            row.clear();
            col = 0;
            rows += 1;
            continue;
        }
        if (byte == '\r') {
            row.clear();
            col = 0;
            continue;
        }
        if (is_continuation(byte)) {
            row.push_back(byte);
            continue;
        }

        if (col >= width) {
            row.clear();
            col = 0;
            rows += 1;
        }
        row.push_back(byte);
        col = byte == '\t' ? std::min(width, ((col / TAB_STOP) + 1) * TAB_STOP)
                           : col + 1;
    }
}
