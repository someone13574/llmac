#!/bin/bash

echo "Running clang-tidy..."
find src -iname '*.cpp' | xargs clang-tidy -p build
