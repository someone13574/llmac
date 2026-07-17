#!/bin/bash

echo "Formatting project..."
find src -type f \( -iname '*.cpp' -o -iname '*.hpp' \) -print0 \
  | xargs -0 clang-format -i
