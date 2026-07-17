#!/bin/bash

set -e

CC=clang CXX=clang++ cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j "$(nproc)"
echo "----------------"
build/src/llmac "$@"
