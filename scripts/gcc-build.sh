#!/bin/bash

set -e

CC=gcc CXX=g++ cmake -G Ninja -B build-gcc -DCMAKE_BUILD_TYPE=Debug
cmake --build build-gcc -j "$(nproc)"
