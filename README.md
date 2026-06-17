# nemotron-asr-core

[![npm version](https://img.shields.io/npm/v/@jeffpeng3/nemotron-asr-core.svg)](https://www.npmjs.com/package/@jeffpeng3/nemotron-asr-core)

Streaming in-browser speech recognition using NVIDIA Nemotron 3.5
(FastConformer-RNNT) via `onnxruntime-web` on WebGPU. Fully client-side
— no server needed. Supports live mic capture, file transcription, and
5 latency-accuracy profiles from a single INT4-quantized encoder.

```
npm install @jeffpeng3/nemotron-asr-core
```

## Live Demo

A demo app is included under `example/`. Run it with Vite:

```bash
npm install
npm run dev
```

Then open the URL shown in the terminal (usually `http://localhost:5173`).

## Features

- **Single INT4 encoder** — all 5 profiles share one `encoder.onnx` (~462 KB
  model + ~733 MB weights, asymmetric INT4 quantized)
- **5 latency profiles** — 80 ms to 1120 ms via `freeDimensionOverrides`
- **Streaming & full-audio** — mic capture or file transcription
- **Greedy + beam search** (configurable 1–5 beams)
- **blankPenalty=0.5 by default** — suppresses blank-dominated output for
  both greedy and beam search
- **Multilingual** — auto-detect or pick from 20+ language IDs
- **WebGPU encoder** — decoder/joint on WASM (CPU); single-threaded by
  default (`numThreads=1`, overridable)

## Usage

```js
import { AsrEngine } from "@jeffpeng3/nemotron-asr-core";

// Callbacks for UI updates
const engine = new AsrEngine({
  progress(label, loaded, total, cached) {
    console.log(`${label}: ${loaded}/${total}`);
  },
  status(detail) {
    console.log(detail);
  },
  partial(text, lang, progress) {
    console.log(`partial (${(progress * 100).toFixed(0)}%): ${text}`);
  },
  ep(encoder, provider, note) {
    console.log(`encoder: ${provider}${note ? ` (${note})` : ""}`);
  },
});

// Download model weights (~863 MB total, cached on-device)
await engine.init();

// ── Full audio transcription ──
const result = await engine.transcribe(samples, 101);
// samples: Float32Array of 16 kHz PCM
// 101 = auto-detect language

console.log(result.text);
// { text: "hello world <en-US>", lang: "en-US", tokens: 12, timing: { ... } }

// ── Streaming (mic / tab capture) ──
const session = engine.session(101);

// push chunks as they arrive
for (const chunk of audioChunks) {
  const partial = await session.feed(chunk);
  if (partial) console.log(partial.text);
}

const final = await session.end();
console.log(final.text);

// ── Benchmark ──
const results = await engine.benchmark({ duration: 10 });
for (const r of results) {
  console.log(`${r.profile} RTF ${r.rtf.toFixed(3)}`);
}
```

## Latency Profiles

All profiles use the same `encoder.onnx` with a dynamic time dimension
pinned at runtime via `freeDimensionOverrides`.

| Profile   | Latency | Encoder Frames |
|-----------|---------|----------------|
| `TURBO`   | 80 ms   | 17             |
| `FAST`    | 160 ms  | 25             |
| `BALANCED`| 320 ms  | 33             |
| `NORMAL`  | 560 ms  | 49             |
| `HIGH`    | 1120 ms | 65             |

Lower latency = fewer context frames = less accurate. Choose the profile
that fits your use case.

```js
await engine.switchProfile("HIGH");  // highest accuracy
```

## API

### `AsrEngine.preload(onProgress?)`

Pre-download and cache all model files (vocab, encoder, decoder, joint) before
creating an engine instance. Subsequent `init()` calls will find files already
in cache and skip network. Useful for showing a download progress screen early
in the app lifecycle.

```js
await AsrEngine.preload((label, loaded, total, cached) => {
  console.log(`${label}: ${loaded}/${total}`);
});
// Now engine.init() will be near-instant
const engine = new AsrEngine(callbacks);
await engine.init();
```

**Arguments:**

| Argument | Type | Description |
|----------|------|-------------|
| `onProgress` | `(label, loaded, total, cached?) => void` | Optional download progress callback |

### `new AsrEngine(callbacks?, options?)`

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `profile` | `"NORMAL"` | Initial latency profile |
| `beamWidth` | `1` | Beam search width (1 = greedy) |
| `blankPenalty` | `0.5` | Subtract from blank logit (both greedy & beam) |
| `blankTop2Threshold` | `0.3` | Secondary blank suppression threshold |
| `numThreads` | `1` | WASM threads for decoder/joint |
| `wasmPaths` | CDN | Custom path for onnxruntime-web WASM files. Set to a local directory when bundling in a Chrome extension or offline environment to avoid CSP issues. |

**Callbacks:**

| Callback | Arguments | Description |
|----------|-----------|-------------|
| `progress` | `(label, loaded, total, cached?)` | Model download progress |
| `status` | `(detail)` | Status messages |
| `partial` | `(text, lang, progress?)` | Partial transcription result |
| `ep` | `(isEncoder, provider, note?)` | Execution provider selection |

### `session(langId)` → `Session`

Create a streaming session.

| Method | Returns | Description |
|--------|---------|-------------|
| `feed(samples)` | `{text, lang} \| null` | Push audio chunk, get partial result |
| `end()` | `{text, lang, tokens, timing}` | Finalize and get complete result |

### `benchmark(opts?)` → `BenchmarkProfileResult[]`

Test all profiles and return RTF (Real-Time Factor) measurements.

**Options:** `{ profiles?, duration?, langId?, warmup?, forceAll?, samples? }`

### `switchProfile(name)`

Switch latency profile at runtime (reloads encoder session).

### `clearCache()`

Remove cached model weights and reset all sessions. Next `init()`
re-downloads from Hugging Face.

### `getPerfStats()` → `Record<string, {ms, calls, avg}>`

Per-operation performance statistics (encoderStep, decoderStep, jointArgmax).

## Language IDs

Pass `101` for auto-detect. Use `0` for English, `4` for Chinese, etc.
See `LANG_TO_ID` and `langId()` exports for the full list.

## Architecture

```
app.js (main thread)  ←→  worker.js (Web Worker)  ←→  Hugging Face / Cache API
                            └── AsrEngine
                                  ├── Mel filterbank + FFT
                                  ├── Encoder (WebGPU — required)
                                  ├── Decoder + Joint (WASM, single-thread)
                                  └── RNN-T greedy / beam search
```

- Model weights (~863 MB) are fetched once from Hugging Face and cached
  via the Cache API with a versioned cache name.
- All inference runs off the main thread via a Web Worker.
- **Encoder requires WebGPU** (D3D12 on Windows, Vulkan on Linux, Metal
  on macOS). Decoder + joint run on WASM (CPU).
- WASM multi-threading (`numThreads > 1`) requires cross-origin isolation
  headers (`Cross-Origin-Opener-Policy` + `Cross-Origin-Embedder-Policy`).
  Disabled by default — pass `{ numThreads: 11 }` to enable if your
  deployment supports it.

## Requirements

- **Browser**: Chrome 113+ / Edge 113+ with WebGPU. Safari 18+ on iOS.
  Firefox Nightly with `dom.webgpu.enabled`.
- **GPU**: ~750 MB of GPU-accessible memory. Integrated GPUs may page
  weights over PCIe (slower).
- **Network**: Model weights (~863 MB) downloaded once, cached locally.
