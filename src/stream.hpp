#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "ac.hpp"

class HexStream {
  public:
    [[nodiscard]] ac::BitSink sink();

    void finish();

  private:
    void push(std::uint32_t bit);

    std::uint32_t carry = 0;
    std::size_t held = 0;
};

class TextStream {
  public:
    using Render = std::function<std::string(ac::Symbol)>;

    explicit TextStream(Render render);

    [[nodiscard]] ac::Sink sink();

    // Flushes any held-back text, ends the line and reports deferred notes.
    void finish();

  private:
    void emit(ac::Symbol symbol);
    void rewind(std::size_t count);
    void commit();
    void note(std::string_view message);

    void write(std::string_view text);
    void advance(std::string_view text);

    Render render;
    bool interactive;
    std::size_t width;

    std::vector<std::string> pieces;
    std::size_t committed = 0;
    std::vector<std::string> notes;

    std::string row;
    std::string start_row;
    std::size_t col = 0;
    std::size_t rows = 0;
};
