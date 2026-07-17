#!/bin/bash

set -e

scripts/format.sh
scripts/gcc-build.sh
scripts/dev-build.sh
scripts/tidy.sh
