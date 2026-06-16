# Encoder ONNX Export Comparison

**Date:** 2026-06-16
**Base model:** NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b
**Profile:** NORMAL (NEW_FRAMES=56, PRE_ENCODE_CACHE=9, ENC_IN=65)
**Export:** `dynamo=True`, opset=21, `Dim("time", min=17, max=121)`
**Input:** 56-frame random noise / sine sweep / zeros → padded to 65 with 9 leading zeros
**Reference:** NeMo float32 (first chunk, `cache_last_channel_len=0`)
**Metrics:** Truncated to `min(NeMo_len, ONNX_len)` frames

## Results (single chunk vs NeMo float32)

| Variant | Quant | bs | sym | acc | Size | avg cos↑ | avg MAE↓ |
|---------|-------|----|-----|-----|------|----------|----------|
| F32 | none | — | — | — | 4763MB | 1.000 | 0.000 |
| **ASYM** | **INT4** | **16** | **No** | **4** | **734MB** | **0.967** | **0.032** |
| B | INT4 | 16 | Yes | 4 | 717MB | 0.884 | 0.041 |
| D | INT4 | 16 | Yes | 5 | 717MB | 0.886 | 0.042 |
| A | INT4 | 32 | Yes | 4 | 653MB | 0.633 | 0.054 |
| C | INT4 | 32 | Yes | 5 | 653MB | 0.627 | 0.054 |
| OFF | INT4 | ? | ? | ? | 661MB | 0.549 | 0.078 |

## Multi-chunk error accumulation
(Asymmetric bs=16, measured in Python, each chunk has different random audio)

Single chunk cos=0.967. Multi-chunk (10 steps) with cache error accumulation shows avg cos~0.65 — errors compound but decoder must be robust to this.

## Key Insights

1. **Dynamo export is correct** — F32 matches NeMo at machine epsilon (cos=1.000)
2. **block_size=16 is strongly preferred** over block_size=32 (cos 0.88 vs 0.63).
3. **Asymmetric quantization (is_symmetric=False) is far better** — cos jumps from 0.884 to 0.967 with the same block_size. The model's weight distributions are clearly asymmetric.
4. **Multi-chunk error accumulation is significant** — cos drops from 0.967 to ~0.65 over 10 chunks as cache errors compound.
5. **Official model (OFF)** underperforms all our quantized variants.

## Chosen Config

**Asymmetric**: `block_size=16, accuracy_level=4, is_symmetric=False`
- Single chunk cos=0.967 (vs 0.884 symmetric)
- 734MB (only 2% larger than symmetric 717MB)
- Uploaded to HF as `encoder.onnx` + `encoder.onnx.data`
