# llmac

Lossless text compression using an LLM as the probability model for an
arithmetic coder.

This can compress natural text or code at very high compression rates, beating most traditional compressors (gzip, bzip2, xz, zstd, brotli) by a fairly wide margin, assuming both the encoder and decoder already have access to the model. It does this by leveraging the predictive capacity of large language models, using their next-token probabilities as probabilties for [Adaptive Arithmetic Coding](https://en.wikipedia.org/wiki/Arithmetic_coding#Adaptive_arithmetic_coding). When used without error correction, this can achieve very low overhead (<0.2% on texts longer than a few tokens) over the theoretical minimum encoding length derived from the distribution's entropy.

Since small differences in the output logits are possible due to floating-point operations, there is also an error-correction mode which frames the code in blocks and includes CRC checkpoints and verification. This allows small differences to be detected and then repaired in case the quantized probability crosses a boundary. Without this, any difference in the quantized probability would amplify each token, eventually causing a complete misprediction. Note that this isn't normally necessary due to the way which llama.cpp has been compiled. Error correction introduces a decent amount of overhead, varying from 1%-15% depending on text length and content (since error correction is done per-block, which is a fixed amount of tokens, the more the content is compressed, the higher the overhead due to error correction will be).

> [!NOTE]
> Compression and decompression are much slower than traditional compressors due to running a language model on each token. You can expect both encoding and decoding to scale approximately linearly in the long term, while shorter messages will increase quadratically. Encoding runs inference in batches of 512 tokens while decoding is unbatched, making it slower.

## Build

Needs CMake ≥ 3.25, Ninja, and a C++23 compiler. llama.cpp (tag `b10063`)
is fetched automatically the first time CMake runs.

```sh
# Or you can run scripts/run.sh to build and run
CC=clang CXX=clang++ cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

## Usage

```
usage: llmac <encode|decode> [-f] [-r] [-m <path>] <input>
  encode|decode        compress text to hex / hex back to text
  -f, --file           read the input from the file at the given path
  -r, --raw            single unblocked stream, no error detection/repair
  -m, --model <path>   gguf model to use (default: models/Qwen3-0.6B-Q4_K_M.gguf)
```

```sh
# encode: hex payload on stdout, stats on stderr
build/src/llmac encode "The quick brown fox jumped over the lazy dog."
build/src/llmac encode -f notes.txt > notes.hex

# decode it back (must use the same model and the same -r choice)
build/src/llmac decode -f notes.hex

# use a different model / the raw wire format
build/src/llmac encode -r -m models/Qwen3.5-9B-Q4_K_M.gguf -f notes.txt
```

Encoding reports the payload size in bits, plus the theoretical minimum
(sum of -log2(p) over the quantized probabilities) so the coding overhead is
visible.

> [!WARNING]
> Decoding must be done with the _exact_ same model file and choice of `-r` flag, otherwise it will generate random text (equivalent to sampling the model with a temperature of 1.0 and no sampler), or if run without `-r` it will try and fail to repair the failing block.

## Benchmark

Compression compared against general-purpose compressors at their maximum
settings: `gzip -9` (1.14), `bzip2 -9` (1.0.8), `xz -9e` (5.8.3),
`zstd --ultra -22` (1.5.7), `brotli -q 11` (1.2.0).

The test texts (in `test-texts/`, except the code ones) were chosen so that
they cannot have appeared in any of the models' training data. Note that `pi10000` and `quick-brown-fox` are exceptions to this, and have likely appear a lot in training data.

| Text            | Domain          | Bytes  | Source                                                                                                                                                                                                                                                                        |
| --------------- | --------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| quick-brown-fox | pangram         | 46     | Pre-existing repo test file. **Certainly in training data.**                                                                                                                                                                                                                  |
| random-base64   | random          | 693    | base64 of 512 bytes of `/dev/urandom`. Incompressible.                                                                                                                                                                                                                        |
| code-small      | C++             | 1,103  | `src/stream.hpp` from this repo at commit `2a09165`.                                                                                                                                                                                                                          |
| wiki-short      | medical text    | 1,246  | Full plaintext of Wikipedia's ["Active stand test"](https://en.wikipedia.org/wiki/Active_stand_test), an article created July 21, 2026, the day of the benchmark.                                                                                                             |
| arxiv-abstract  | scientific text | 1,759  | Title + abstract of ["A Self-Consistent 3D Hydrodynamic Model for Helium Transit Signatures in Evaporating Hot Jupiters"](https://arxiv.org/abs/2607.18193), submitted to arXiv July 20, 2026.                                                                                |
| news-article    | journalism      | 3,363  | Body text of a Guardian article from July 21, 2026: ["Healthy diet too expensive for one in three people globally, UN report finds"](https://www.theguardian.com/global-development/2026/jul/21/healthy-diet-too-expensive-for-one-in-three-people-globally-un-report-finds). |
| pi10000         | digits          | 10,002 | First 10,000 digits of π. **Certainly in training data**, though incompressible unless memorized.                                                                                                                                                                             |
| wiki-long       | sports prose    | 29,029 | Full plaintext of Wikipedia's ["2026 FIFA World Cup final"](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_final). The match was played July 19, 2026.                                                                                                                     |
| code-all        | C++/CMake       | 49,931 | All ten source files of this repo at commit `2a09165`, concatenated in a fixed order with `// ===== <path> =====` separator lines.                                                                                                                                            |

Models (all Q4_K_M GGUF, CPU inference via llama.cpp `b10063` on an
AMD Ryzen 7 PRO 8840HS using 8 threads): **Qwen3-0.6B**, **SmolLM3-3B-Base**,
**Qwen3.5-9B**.

### Compressed size in bytes, with error-correction (lower is better, best per row in bold)

| Text            | Original | gzip -9 | bzip2 -9 | xz -9e | zstd -22 | brotli -q 11 | llmac Qwen3-0.6B | llmac SmolLM3-3B | llmac Qwen3.5-9B |
| --------------- | -------- | ------- | -------- | ------ | -------- | ------------ | ---------------- | ---------------- | ---------------- |
| quick-brown-fox | 46       | 85      | 84       | 112    | 59       | 51           | 13.5             | 12               | **9.5**          |
| random-base64   | 693      | 591     | 648      | 676    | 576      | **547**      | 610              | 573.5            | 575.5            |
| code-small      | 1,103    | 472     | 517      | 540    | 463      | 388          | 127.5            | **91.5**         | 94.5             |
| wiki-short      | 1,246    | 635     | 669      | 728    | 620      | 460          | 172              | 125.5            | **115.5**        |
| arxiv-abstract  | 1,759    | 876     | 889      | 956    | 843      | 642          | 233              | 191              | **176**          |
| news-article    | 3,363    | 1,662   | 1,625    | 1,740  | 1,630    | 1,163        | 414              | **278.5**        | 332              |
| pi10000         | 10,002   | 5,119   | 4,383    | 4,548  | 4,417    | 4,375        | 4,180            | **2,522.5**      | 3,114            |
| wiki-long       | 29,029   | 11,317  | 10,103   | 10,716 | 10,782   | 9,094        | 3,804.5          | 2,532.5          | **2,498.5**      |
| code-all        | 49,931   | 11,846  | 10,760   | 11,072 | 11,226   | 10,563       | 3,633            | 2,547.5          | **2,387.5**      |

### Compression, percent of original size removed, raw mode (non-blocked) in brackets (higher is better, best per row in bold; negative means the output grew)

| Text            | gzip -9 | bzip2 -9 | xz -9e  | zstd -22 | brotli -q 11 | Qwen3-0.6B    | SmolLM3-3B            | Qwen3.5-9B            |
| --------------- | ------- | -------- | ------- | -------- | ------------ | ------------- | --------------------- | --------------------- |
| quick-brown-fox | -84.8%  | -82.6%   | -143.5% | -28.3%   | -10.9%       | 70.7% [74.7%] | 73.9% [78.0%]         | **79.3%** [**83.4%**] |
| random-base64   | 14.7%   | 6.5%     | 2.5%    | 16.9%    | **21.1%**    | 12.0% [13.9%] | 17.2% [18.5%]         | 17.0% [18.2%]         |
| code-small      | 57.2%   | 53.1%    | 51.0%   | 58.0%    | 64.8%        | 88.4% [89.1%] | **91.7%** [**92.4%**] | 91.4% [92.1%]         |
| wiki-short      | 49.0%   | 46.3%    | 41.6%   | 50.2%    | 63.1%        | 86.2% [86.8%] | 89.9% [90.5%]         | **90.7%** [**91.3%**] |
| arxiv-abstract  | 50.2%   | 49.5%    | 45.7%   | 52.1%    | 63.5%        | 86.8% [87.2%] | 89.1% [89.6%]         | **90.0%** [**90.4%**] |
| news-article    | 50.6%   | 51.7%    | 48.3%   | 51.5%    | 65.4%        | 87.7% [88.1%] | **91.7%** [**92.1%**] | 90.1% [90.6%]         |
| pi10000         | 48.8%   | 56.2%    | 54.5%   | 55.8%    | 56.3%        | 58.2% [60.4%] | **74.8%** [**75.5%**] | 68.9% [71.1%]         |
| wiki-long       | 61.0%   | 65.2%    | 63.1%   | 62.9%    | 68.7%        | 86.9% [87.4%] | 91.3% [91.8%]         | **91.4%** [**91.9%**] |
| code-all        | 76.3%   | 78.5%    | 77.8%   | 77.5%    | 78.8%        | 92.7% [93.3%] | 94.9% [95.5%]         | **95.2%** [**95.9%**] |

### llmac breakdown per model (bits)

_Theoretical_ is from the quantized model probabilities alone. _Raw_ (`-r`) is the bit length of the non-blocked arithmetic-coded stream with no error detection. _Default_ adds the blocked wire format, including CRC checkpoints, final CRC, and padding. _Bits/byte_ is raw bits per byte of original text. Cells with `-` were not measured.

#### Qwen3-0.6B

| Text            | Tokens | Theoretical | Raw    | AC overhead | Default | Framing overhead | Bits/byte | Encode time | Decode time |
| --------------- | ------ | ----------- | ------ | ----------- | ------- | ---------------- | --------- | ----------- | ----------- |
| quick-brown-fox | 11     | 92.0        | 93     | 1.09%       | 108     | 16.13%           | 2.02      | 0.3 s       | 0 s         |
| random-base64   | 513    | 4,772.5     | 4,774  | 0.03%       | 4,880   | 2.22%            | 6.89      | 2.9 s       | 7 s         |
| code-small      | 282    | 961.8       | 963    | 0.12%       | 1,020   | 5.92%            | 0.87      | 1.7 s       | 4 s         |
| wiki-short      | 271    | 1,317.2     | 1,318  | 0.06%       | 1,376   | 4.40%            | 1.06      | 1.8 s       | 4 s         |
| arxiv-abstract  | 353    | 1,799.7     | 1,800  | 0.02%       | 1,864   | 3.56%            | 1.02      | 2.2 s       | 5 s         |
| news-article    | 726    | 3,194.0     | 3,195  | 0.03%       | 3,312   | 3.66%            | 0.95      | 4.3 s       | 10 s        |
| pi10000         | 10,003 | 31,671.2    | 31,672 | 0.00%       | 33,440  | 5.58%            | 3.17      | 64.0 s      | -           |
| wiki-long       | 6,701  | 29,253.5    | 29,254 | 0.00%       | 30,436  | 4.04%            | 1.01      | 43.0 s      | -           |
| code-all        | 13,160 | 26,751.4    | 26,752 | 0.00%       | 29,064  | 8.64%            | 0.54      | 81.2 s      | -           |

#### SmolLM3-3B

| Text            | Tokens | Theoretical | Raw    | AC overhead | Default | Framing overhead | Bits/byte | Encode time | Decode time |
| --------------- | ------ | ----------- | ------ | ----------- | ------- | ---------------- | --------- | ----------- | ----------- |
| quick-brown-fox | 11     | 79.8        | 81     | 1.50%       | 96      | 18.52%           | 1.76      | 1.2 s       | 1 s         |
| random-base64   | 495    | 4,518.1     | 4,519  | 0.02%       | 4,588   | 1.53%            | 6.52      | 10.5 s      | 20 s        |
| code-small      | 280    | 674.4       | 675    | 0.09%       | 732     | 8.44%            | 0.61      | 5.9 s       | 11 s        |
| wiki-short      | 270    | 947.0       | 948    | 0.11%       | 1,004   | 5.91%            | 0.76      | 6.2 s       | 11 s        |
| arxiv-abstract  | 350    | 1,463.3     | 1,465  | 0.12%       | 1,528   | 4.30%            | 0.83      | 7.8 s       | 14 s        |
| news-article    | 700    | 2,114.3     | 2,115  | 0.03%       | 2,228   | 5.34%            | 0.63      | 14.5 s      | 28 s        |
| pi10000         | 3,337  | 19,579.8    | 19,581 | 0.01%       | 20,180  | 3.06%            | 1.96      | 72.2 s      | -           |
| wiki-long       | 6,433  | 19,114.4    | 19,115 | 0.00%       | 20,260  | 5.99%            | 0.66      | 143.2 s     | -           |
| code-all        | 12,883 | 18,110.5    | 18,111 | 0.00%       | 20,380  | 12.53%           | 0.36      | 283.3 s     | -           |

#### Qwen3.5-9B

| Text            | Tokens | Theoretical | Raw    | AC overhead | Default | Framing overhead | Bits/byte | Encode time | Decode time |
| --------------- | ------ | ----------- | ------ | ----------- | ------- | ---------------- | --------- | ----------- | ----------- |
| quick-brown-fox | 12     | 60.0        | 61     | 1.67%       | 76      | 24.59%           | 1.33      | 1.4 s       | 2 s         |
| random-base64   | 510    | 4,531.8     | 4,533  | 0.03%       | 4,604   | 1.57%            | 6.54      | 26.7 s      | 54 s        |
| code-small      | 323    | 693.4       | 694    | 0.09%       | 756     | 8.93%            | 0.63      | 17.0 s      | 35 s        |
| wiki-short      | 271    | 864.3       | 866    | 0.20%       | 924     | 6.70%            | 0.70      | 14.5 s      | 29 s        |
| arxiv-abstract  | 353    | 1,345.2     | 1,346  | 0.06%       | 1,408   | 4.61%            | 0.77      | 18.8 s      | 38 s        |
| news-article    | 743    | 2,539.6     | 2,540  | 0.02%       | 2,656   | 4.57%            | 0.76      | 37.9 s      | 79 s        |
| pi10000         | 10,003 | 23,140.9    | 23,142 | 0.00%       | 24,912  | 7.65%            | 2.31      | 535.2 s     | -           |
| wiki-long       | 6,630  | 18,835.4    | 18,836 | 0.00%       | 19,988  | 6.12%            | 0.65      | 348.9 s     | -           |
| code-all        | 14,694 | 16,503.8    | 16,505 | 0.01%       | 19,100  | 15.72%           | 0.33      | 767.8 s     | -           |
