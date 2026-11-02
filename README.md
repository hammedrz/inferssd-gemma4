<h1 align="center">Large MoE models, powered by your SSD</h1>

<p align="center">
  <a href="https://wici.ai">
    <img src="docs/assets/wicissdinfer.png" width="600" alt="InferSSD">
  </a>
</p>

<p align="center">
  <a href="https://wici.ai/inferssd"><b>Website</b></a> &nbsp;·&nbsp;
  <a href="https://wici.ai/research"><b>Blog</b></a> &nbsp;·&nbsp;
  <a href="https://join.slack.com/t/wici-ai/shared_invite/zt-4f2e7ob2s-9nUg35Phqyqt4EI6qUjw5w"><b>Slack</b></a> &nbsp;·&nbsp;
  <a href="LICENSE">Apache-2.0</a>
</p>
<p align="center">macOS / Metal &nbsp;·&nbsp; Linux / CUDA &nbsp;·&nbsp; Windows (in progress)</p>

> [!NOTE]
> **Unofficial fork.** This is [hammedrz/inferssd-gemma4](https://github.com/hammedrz/inferssd-gemma4), a fork of
> [wici-ai/inferssd](https://github.com/wici-ai/inferssd) that adds smaller mixture-of-experts models. It is not
> affiliated with or endorsed by WiCi AI. Below this section the README is upstream's, with the install command
> pointed at this fork.

## About this fork

This fork streams the routed experts of two smaller MoE architectures from the SSD, in addition to the models
upstream supports:

| Model | GGUF architecture | Parameters | Tested GGUF |
| --- | --- | --- | --- |
| DeepSeek-V2-Lite-Chat | `deepseek2` | 15.7B total / 2.4B active | [mradermacher · Q8_0](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) |
| Gemma 4 26B-A4B-it | `gemma4` | 25.8B total / about 4B active | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |

Dense models of the same architectures, such as Gemma 4 E4B, have no routed experts and are not supported.
Run a downloaded file with `inferssd cli -m /path/to/model.gguf` or `inferssd serve -m /path/to/model.gguf`.

### Changes from upstream

- **CPU expert executor:** GELU-gated experts (`GEGLU`, computed with ggml's own kernel, so values match the
  graph) and an optional per-expert down-projection scale, applied in the graph's order: the down output times
  the scale, then the routing weight. The CPU backend initializes ggml's GELU table when an executor is created.
- **Merged gate/up experts:** Gemma 4 stores each expert's gate and up rows in one `ffn_gate_up_exps` tensor. The
  engine describes its two halves to the executor as separate gate and up tensors, and the GGUF alias index
  splits each expert's bytes the same way.
- **Model support:** the model adapter accepts `deepseek2` and `gemma4`; the launcher maps both to a `gguf`
  profile that keeps the chat template embedded in the GGUF.
- **`inferssd cli` and `serve`:** an optional `--memory-gib` caps the memory limit; without it they use all of
  physical memory as upstream does.
- **Installer:** builds this fork, and on Linux without a CUDA toolkit builds the CPU-only engine (`linux-cpu`)
  instead of stopping.

Modified files: `engine/ggml/src/ggml-cpu/ssd/cpu-sink.{h,cpp}`, `engine/ggml/src/ggml-cpu/ssd/cpu-prefill.cpp`,
`engine/ggml/src/ggml-cpu/ssd/cpu-cache-executor.inc`, `engine/src/llama-graph.cpp`, `engine/src/llama-context.cpp`,
`engine/src/ssd/cpu-runtime.h`, `engine/src/ssd/model-adapter.{h,cpp}`, `engine/tests/CMakeLists.txt`,
`engine/tests/ssd/test-kimi-cpu-sink.cpp`, `engine/tests/ssd/test-model-adapter.cpp`,
`framework/src/inferssd/app/command.py`, `framework/src/inferssd/launch/typed.py`,
`framework/src/inferssd/source/{experts,gguf_alias}.py`, `framework/src/inferssd/strategy/inputs.py`,
`scripts/install.sh`, `README.md` and `NOTICE`. The git history records each change.

### Validation

On a Linux laptop with an Intel Core i5-1145G7 (4 cores, 8 threads), 16 GB of RAM, a Crucial P310 NVMe SSD
and the CPU-only build:

- **Unit tests:** the Gemma 4 expert test (`test-kimi-cpu-sink --gemma4-geglu-test`) matches ggml's MoE graph bit
  for bit in decode and in 3- and 65-token grouped prefill, for Q4_K/Q5_1 and Q8_0 experts, with and without
  per-expert scales. The existing CPU executor suites, the model adapter test and the framework tests pass.
- **DeepSeek-V2-Lite-Chat Q8_0:** 48 of 48 greedy tokens and their log probabilities were identical to the same
  build without SSD experts. Generation ran at 7.2 token/s against 1.6 token/s without SSD experts (one short
  run each, not a benchmark).
- **Gemma 4 26B-A4B UD-Q4_K_XL:** answers are correct and fluent. Before the per-expert scale moved into the
  executor, the only difference from llama.cpp was that scale's rounding order. Applying the same order in
  llama.cpp made all 48 tokens identical. The current version has not yet been compared on the full model.

Only the Linux CPU build was tested. A CUDA GPU expert tier refuses GELU-gated experts and per-expert scales
rather than computing them; macOS was not tested.

**Run models larger than your RAM or VRAM.** InferSSD builds on [llama.cpp](https://github.com/ggml-org/llama.cpp) to run large mixture-of-experts (MoE) models from SSD, using standard GGUF files, with browser chat and an OpenAI-compatible API. It currently supports **DeepSeek-V4.1-Flash-1017** and **Muse-Spark-Open**, with more models coming.

## Models

These are the tested GGUF files, community conversions of the named model releases. DeepSeek's Q8_0 keeps the official bit widths: routed experts stay native 4-bit MXFP4, and FP8 tensors become 8-bit Q8_0. See [weight-format details](docs/benchmark.md#checkpoint-formats).

| Model | Parameter count | GGUF reference |
| --- | --- | --- |
| DeepSeek-V4.1-Flash-1017 | 763B total / 8–16B active | [mrfakename · Q8_0](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) |
| Muse-Spark-Open | 816B total / 42B active | [Unsloth · UD-Q4_K_XL](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF) |

## Performance benchmarks

**4K prompt tier · 256 output tokens · decode measured over tokens 129–256.** All rates are in **token/s**; higher is faster. The multiple next to each **InferSSD** rate is its speedup over llama.cpp on the same hardware. On the MacBook Pro, Colibri runs the official Safetensors on CPU; llama.cpp and InferSSD run the GGUF with Metal. See [measurement details](docs/benchmark.md).

### Test hardware

| Hardware | CPU (cores) | GPU | Memory | SSD |
| --- | --- | --- | --- | --- |
| MacBook&nbsp;Pro | Apple&nbsp;M5&nbsp;Max&nbsp;(18) | 40-core | 48&nbsp;GB&nbsp;unified | 2&nbsp;TB&nbsp;internal |
| RTX&nbsp;5060&nbsp;Ti | Intel&nbsp;Core&nbsp;3&nbsp;100U&nbsp;(6) | RTX&nbsp;5060&nbsp;Ti | 8&nbsp;GB&nbsp;+&nbsp;16&nbsp;GB&nbsp;VRAM | 2&nbsp;TB&nbsp;PCIe&nbsp;4&nbsp;×4 |
| RTX&nbsp;5090 | Intel&nbsp;Core&nbsp;Ultra&nbsp;5&nbsp;230F&nbsp;(10) | RTX&nbsp;5090 | 32&nbsp;GB&nbsp;+&nbsp;32&nbsp;GB&nbsp;VRAM | 2&nbsp;TB&nbsp;PCIe&nbsp;5&nbsp;×4 |
| DGX&nbsp;Spark | NVIDIA&nbsp;GB10&nbsp;(20) | Blackwell | 128&nbsp;GB&nbsp;unified | 4&nbsp;TB&nbsp;PCIe&nbsp;5&nbsp;×4 |

Unified memory is shared by the CPU and GPU. SSD capacities are nominal. See [hardware details](docs/benchmark.md#hardware-details) for SSD models, CPU core layouts, and configuration notes.

### DeepSeek-V4.1-Flash-1017

<table>
<thead>
<tr><th rowspan="2">Hardware</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro</td><td align="right">5.929</td><td align="right">0.708</td><td align="right"><b>28.766</b>&nbsp;(4.85×)</td><td align="right">0.873</td><td align="right">0.437</td><td align="right"><b>2.874</b>&nbsp;(3.29×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">6.950</td><td align="right">7.020</td><td align="right"><b>75.100</b>&nbsp;(10.81×)</td><td align="right">0.242</td><td align="right">0.900</td><td align="right"><b>2.550</b>&nbsp;(10.54×)</td></tr>
<tr><td>RTX 5090</td><td align="right">8.060</td><td align="right">39.210</td><td align="right"><b>164.500</b>&nbsp;(20.41×)</td><td align="right">0.867</td><td align="right">1.878</td><td align="right"><b>7.270</b>&nbsp;(8.39×)</td></tr>
<tr><td>DGX Spark</td><td align="right">2.730</td><td align="right">0.826</td><td align="right"><b>127.500</b>&nbsp;(46.70×)</td><td align="right">1.608</td><td align="right">0.356</td><td align="right"><b>7.260</b>&nbsp;(4.51×)</td></tr>
</tbody>
</table>

### Muse-Spark-Open

<table>
<thead>
<tr><th rowspan="2">Hardware</th><th colspan="3">Prefill (token/s)</th><th colspan="3">Decode (token/s)</th></tr>
<tr><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th><th>llama.cpp</th><th>Colibri</th><th>InferSSD</th></tr>
</thead>
<tbody>
<tr><td>MacBook&nbsp;Pro</td><td align="right">3.140</td><td align="right">N/S</td><td align="right"><b>14.503</b>&nbsp;(4.62×)</td><td align="right">0.153</td><td align="right">N/S</td><td align="right"><b>1.097</b>&nbsp;(7.17×)</td></tr>
<tr><td>RTX 5060 Ti</td><td align="right">3.410</td><td align="right">N/S</td><td align="right"><b>52.100</b>&nbsp;(15.28×)</td><td align="right">0.059</td><td align="right">N/S</td><td align="right"><b>0.219</b>&nbsp;(3.71×)</td></tr>
<tr><td>RTX 5090</td><td align="right">4.000</td><td align="right">N/S</td><td align="right"><b>106.600</b>&nbsp;(26.65×)</td><td align="right">0.155</td><td align="right">N/S</td><td align="right"><b>1.520</b>&nbsp;(9.81×)</td></tr>
<tr><td>DGX Spark</td><td align="right">1.740</td><td align="right">N/S</td><td align="right"><b>83.900</b>&nbsp;(48.22×)</td><td align="right">0.120</td><td align="right">N/S</td><td align="right"><b>2.140</b>&nbsp;(17.83×)</td></tr>
</tbody>
</table>

**N/S:** Muse is not currently supported by Colibri.

## Get started

Install on Apple Silicon macOS or on Linux with an NVIDIA GPU:

```sh
curl -fsSL https://raw.githubusercontent.com/hammedrz/inferssd-gemma4/main/scripts/install.sh | sh
```

InferSSD currently installs from source, so the engine is compiled for your machine's CPU and GPU; prebuilt packages are not available yet. Building takes about 2 to 12 minutes, depending on the machine.

The installer works in a private environment and uses no `sudo`. It needs Git, Python 3.9 or later with `venv` (on Debian and Ubuntu, the `python3-venv` package), and a C/C++ compiler. macOS also needs the Xcode Command Line Tools. Linux needs systemd with cgroup v2; with the CUDA toolkit it builds the CUDA engine, which also needs `liburing` headers, and without it this fork builds the CPU-only engine. Follow the printed PATH instruction if it shows one.

Then start a model:

```sh
# Chat in your terminal
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

# Or start the web UI and an OpenAI-compatible API at http://127.0.0.1:8080
inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF
```

For Muse, use `unsloth/Muse-Spark-Open-816B-A42B-GGUF`. With `serve`, open `http://127.0.0.1:8080` in your browser, or point any OpenAI-compatible client at `http://127.0.0.1:8080/v1`.

The first run downloads about 470 GiB of model files to `~/.local/share/inferssd/models` and measures your machine. Use `--model-dir /path/on/ssd` to choose the drive, or `-m /path/to/first-shard.gguf` for files you already have. There is no memory setting to choose: InferSSD uses the memory available. Stop with **Ctrl+C**, or `/exit` in chat.

See the [quick start](docs/quickstart.md#one-command-installation) for upgrades and uninstalling.

## Further reading

- [How it works](docs/architecture.md)
- [Benchmark methodology](docs/benchmark.md)

## Contributing

Contributions are welcome, including hardware results and improvements to expert I/O, caching, and batching. See the [contributing guide](CONTRIBUTING.md).

## Acknowledgments and license

Built on [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp), with thanks to their contributors and the model-adaptation community.

InferSSD is licensed under [Apache-2.0](LICENSE); bundled llama.cpp/ggml and other third-party components keep their original licenses and notices; see [NOTICE](NOTICE). Model weights remain subject to their publishers' licenses.
