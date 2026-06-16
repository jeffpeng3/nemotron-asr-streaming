#!/usr/bin/env python3
"""
Full comparison pipeline.

Exports ONNX variants from NeMo → compares each to NeMo reference → prints table.

Usage: uv run python scripts/full_compare.py
"""

import gc, os, sys, time, tempfile, shutil
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

os.environ["CUDA_VISIBLE_DEVICES"] = ""

N_MELS = 128
N_LAYERS = 24
D_MODEL = 1024
CONV_CONTEXT = 8
PRE_ENCODE_CACHE = 9
NUM_PROMPTS = 128
LEFT_CONTEXT = 56
RIGHT_CONTEXT = 13
NEW_FRAMES = 56  # NORMAL profile
ENC_IN = NEW_FRAMES + PRE_ENCODE_CACHE  # = 65

ENC_INPUT_NAMES = ["audio_signal", "length", "cache_last_channel", "cache_last_time",
                   "cache_last_channel_len", "lang_id"]
ENC_OUTPUT_NAMES = ["outputs", "encoded_lengths", "cache_last_channel_next",
                    "cache_last_time_next", "cache_last_channel_len_next"]
OUTDIR = Path("build/compare_test")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── NeMo encoder wrapper ──
class StreamingEncoderWrapper(nn.Module):
    def __init__(self, encoder, prompt_kernel):
        super().__init__()
        self.encoder = encoder
        self.prompt_kernel = prompt_kernel

    def forward(self, audio_signal, length,
                cache_last_channel, cache_last_time, cache_last_channel_len, lang_id):
        audio_signal = audio_signal.transpose(1, 2)
        enc, enc_len, cache_ch_next, cache_tm_next, cache_len_next = \
            self.encoder.forward_for_export(
                audio_signal=audio_signal, length=length,
                cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len)
        enc = enc.transpose(1, 2)
        onehot = F.one_hot(lang_id, num_classes=NUM_PROMPTS).to(enc.dtype)
        prompt = onehot.unsqueeze(1).expand(-1, enc.shape[1], -1)
        enc = self.prompt_kernel(torch.cat([enc, prompt], dim=-1)).to(enc.dtype)
        return enc, enc_len, cache_ch_next, cache_tm_next, cache_len_next


# ── Patch rel_shift ──
from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention
_orig_rel_shift = RelPositionMultiHeadAttention.rel_shift
def _patched_rel_shift(self, x):
    b, h, qlen, pos_len = x.size()
    x = F.pad(x, pad=(1, 0)).reshape(b, h, -1, qlen)
    return x[:, :, 1:].reshape(b, h, qlen, pos_len)


# ── Test signals ──
def make_signals(t):
    noise = torch.randn(1, t, N_MELS)
    try:
        import librosa
        sr, hop = 16000, 160
        n = int(t * hop / sr * sr)
        dur = n / sr
        lin = torch.linspace(0, dur, n)
        f0, f1 = 200, 4000
        sw = torch.sin(2 * torch.pi * (f0 + (f1 - f0) * lin / dur) * lin)
        mel = librosa.feature.melspectrogram(y=sw.numpy(), sr=sr, n_mels=N_MELS,
                                             hop_length=hop, win_length=400, n_fft=512)
        sine = torch.from_numpy(mel).float().unsqueeze(0).transpose(1, 2)
        if sine.shape[1] < t:
            sine = torch.cat([sine, torch.zeros(1, t - sine.shape[1], N_MELS)], dim=1)
        else:
            sine = sine[:, :t, :]
    except ImportError:
        sine = torch.randn(1, t, N_MELS)
    zero = torch.zeros(1, t, N_MELS)
    return {"noise": noise, "sine": sine, "zero": zero}


# ── NeMo loader ──
def load_nemo():
    from nemo.collections.asr.models import ASRModel
    from huggingface_hub import hf_hub_download, list_repo_files
    repo = "nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b"
    nemo_file = [f for f in list_repo_files(repo) if f.endswith(".nemo")][0]
    print(f"  Loading NeMo from {nemo_file} ...")
    t0 = time.time()
    m = ASRModel.restore_from(hf_hub_download(repo_id=repo, filename=nemo_file))
    m = m.cpu().eval()
    print(f"  Done ({time.time()-t0:.0f}s)")
    return m


# ── Run NeMo encoder ──
@torch.no_grad()
def run_nemo_enc(model, audio_mel):
    t = audio_mel.shape[1]
    enc_in = t + PRE_ENCODE_CACHE
    padded = torch.zeros(1, enc_in, N_MELS)
    padded[:, PRE_ENCODE_CACHE:, :] = audio_mel
    wrapper = StreamingEncoderWrapper(model.encoder, model.prompt_kernel).eval()
    out = wrapper(
        padded,
        torch.tensor([enc_in], dtype=torch.int64),
        torch.zeros(1, N_LAYERS, LEFT_CONTEXT, D_MODEL),
        torch.zeros(1, N_LAYERS, D_MODEL, CONV_CONTEXT),
        torch.zeros(1, dtype=torch.int64),
        torch.full((1,), 101, dtype=torch.int64),
    )
    return {
        "outputs": out[0].numpy(),
        "cache_last_channel_next": out[2].numpy(),
        "cache_last_time_next": out[3].numpy(),
    }


# ── Export ONNX ──
def export_onnx(model, name, quant_opts=None):
    path = OUTDIR / f"encoder_{name}.onnx"
    if path.exists():
        print(f"  {name}: ONNX already exists, skipping export")
        return path

    print(f"  Exporting {name} ...")
    wrapper = StreamingEncoderWrapper(model.encoder, model.prompt_kernel).eval()
    dummy = (
        torch.randn(1, 121, N_MELS),
        torch.tensor([121], dtype=torch.int64),
        torch.zeros(1, N_LAYERS, LEFT_CONTEXT, D_MODEL),
        torch.zeros(1, N_LAYERS, D_MODEL, CONV_CONTEXT),
        torch.zeros(1, dtype=torch.int64),
        torch.full((1,), 101, dtype=torch.int64),
    )
    t0 = time.time()
    try:
        r = torch.onnx.export(
            wrapper, dummy, str(path),
            input_names=ENC_INPUT_NAMES, output_names=ENC_OUTPUT_NAMES,
            opset_version=21, dynamo=True,
            dynamic_shapes={"audio_signal": {1: Dim("time", min=17, max=121)},
                            "length": None, "cache_last_channel": None,
                            "cache_last_time": None, "cache_last_channel_len": None,
                            "lang_id": None},
        )
        print(f"    export: {time.time()-t0:.0f}s")
        if hasattr(r, "optimize"):
            r.optimize()
        if hasattr(r, "save"):
            r.save(str(path))
        if not path.exists():
            if hasattr(r, "exported_program"):
                from torch.onnx._internal.exporter._core import ONNXProgram
                ONNXProgram(r.exported_program, {}).save(str(path))
            elif hasattr(r, "model_proto"):
                onnx.save_model(r.model_proto, str(path), save_as_external_data=True,
                                all_tensors_to_one_file=True, location=path.name + ".data")
        else:
            m = onnx.load(str(path), load_external_data=True)
            for k, v in {"profile": "single_rc13", "left_context": str(LEFT_CONTEXT),
                          "right_context": str(RIGHT_CONTEXT)}.items():
                p = m.metadata_props.add()
                p.key, p.value = k, v
            onnx.save_model(m, str(path), save_as_external_data=True,
                            all_tensors_to_one_file=True, location=path.name + ".data")
    except Exception as e:
        print(f"    [FAIL] {e}")
        import traceback; traceback.print_exc()
        return None

    if quant_opts:
        _quantize(path, **quant_opts)
    return path


def _quantize(path, block_size=32, accuracy_level=4, is_symmetric=True):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    print(f"    Quantizing INT4 (bs={block_size}, acc={accuracy_level}, sym={is_symmetric}) ...")
    m = onnx.load(str(path), load_external_data=True)
    for init in m.graph.initializer:
        if len(init.external_data) > 0 or init.data_location != 0:
            init.ClearField('external_data')
            init.ClearField('data_location')
    q = MatMulNBitsQuantizer(model=m, block_size=block_size, is_symmetric=is_symmetric,
                              accuracy_level=accuracy_level)
    q.process()
    data_path = path.with_suffix(path.suffix + ".data")
    if data_path.exists():
        data_path.unlink()
    path.unlink()
    q.model.save_model_to_file(str(path), use_external_data_format=True)
    mb = sum(f.stat().st_size for f in [path, data_path]) / (1024*1024)
    print(f"    done ({mb:.1f} MB)")


# ── Run ONNX ──
def run_onnx(path, audio_np):
    t = audio_np.shape[1]
    enc_in = t + PRE_ENCODE_CACHE
    padded = np.zeros((1, enc_in, N_MELS), dtype=np.float32)
    padded[:, PRE_ENCODE_CACHE:, :] = audio_np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = ["outputs", "cache_last_channel_next", "cache_last_time_next"]
    res = sess.run(names, {
        "audio_signal": padded,
        "length": np.array([enc_in], dtype=np.int64),
        "cache_last_channel": np.zeros((1, N_LAYERS, LEFT_CONTEXT, D_MODEL), dtype=np.float32),
        "cache_last_time": np.zeros((1, N_LAYERS, D_MODEL, CONV_CONTEXT), dtype=np.float32),
        "cache_last_channel_len": np.zeros((1,), dtype=np.int64),
        "lang_id": np.full((1,), 101, dtype=np.int64),
    })
    return dict(zip(names, res))


# ── Metrics ──
def metrics(ref, pred):
    d = ref - pred
    eps = 1e-8
    return {
        "MAE": float(np.mean(np.abs(d))),
        "MSE": float(np.mean(d ** 2)),
        "MAX": float(np.max(np.abs(d))),
        "cos": float(np.dot(ref.ravel(), pred.ravel()) / (
            np.linalg.norm(ref.ravel()) * np.linalg.norm(pred.ravel()) + eps)),
    }


def print_table(rows):
    hdr = f"{'Variant':>8}  {'Signal':>6}  {'Output':>32}  {'MAE':>10}  {'MSE':>10}  {'MAX':>10}  {'cos':>12}  {'Size':>8}"
    sep = "─" * len(hdr)
    print(f"\n{hdr}\n{sep}")
    for v, sig, oname, m, sz in rows:
        print(f"{v:>8}  {sig:>6}  {oname:>32}  {m['MAE']:>10.6f}  {m['MSE']:>10.6f}  {m['MAX']:>10.4f}  {m['cos']:>12.8f}  {sz:>8}")


def download_with_data(repo, filename):
    tmp = Path(tempfile.mktemp(suffix="_onnx"))
    tmp.mkdir(parents=True, exist_ok=True)
    for fn in [filename, filename + ".data"]:
        try:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(repo_id=repo, filename=fn)
            shutil.copy2(src, tmp / fn)
        except Exception:
            pass
    return tmp / filename


# ── Main ──
def main():
    print("=" * 75)
    print(" Full ONNX comparison vs NeMo float32")
    print(f" ENC_IN={ENC_IN} (NORMAL profile, {NEW_FRAMES}+{PRE_ENCODE_CACHE}), signals: noise / sine / zero")
    print("=" * 75)

    RelPositionMultiHeadAttention.rel_shift = _patched_rel_shift

    # ── 1. Load NeMo ──
    print("\n[1] Loading NeMo ...")
    nemo = load_nemo()

    # ── 2. Generate test signals & run NeMo ──
    print("\n[2] Generating test signals & running NeMo encoder ...")
    signals = make_signals(NEW_FRAMES)
    refs = {}
    for sn, mel in signals.items():
        refs[sn] = run_nemo_enc(nemo, mel)
        print(f"  {sn}: outputs={refs[sn]['outputs'].shape}")

    # ── 3. Export all variants ──
    print("\n[3] Exporting ONNX variants ...")
    variants = [
        ("F32", None),
        ("A",   dict(block_size=32, accuracy_level=4)),
        ("B",   dict(block_size=16, accuracy_level=4)),
        ("C",   dict(block_size=32, accuracy_level=5)),
        ("D",   dict(block_size=16, accuracy_level=5)),
        ("E",   dict(block_size=16, accuracy_level=5, is_symmetric=False)),
    ]
    exported = {}
    for vname, qopts in variants:
        p = export_onnx(nemo, vname, qopts)
        if p and p.exists():
            mb = sum(f.stat().st_size for f in [p, p.with_suffix(p.suffix+".data")] if f.exists())/(1024*1024)
            exported[vname] = (p, mb)
        gc.collect()

    # Free NeMo
    del nemo; gc.collect()
    print("  (NeMo freed)")

    # ── 4. Official ONNX ──
    print("\n[4] Fetching official onnx-community encoder ...")
    try:
        off_path = download_with_data("onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4",
                                      "encoder.onnx")
        off_mb = sum(f.stat().st_size for f in [off_path, off_path.with_suffix(off_path.suffix+".data")] if f.exists())/(1024*1024)
        exported["OFF"] = (off_path, off_mb)
        print(f"  official: {off_path.name} ({off_mb:.0f} MB)")
    except Exception as e:
        print(f"  [SKIP official] {e}")

    # ── 5. Compare ──
    print("\n[5] Comparing all variants to NeMo ...")
    out_names = ["outputs", "cache_last_channel_next", "cache_last_time_next"]
    all_rows = []

    for vname, (path, mb) in exported.items():
        for sn in ["noise", "sine", "zero"]:
            onx = run_onnx(path, signals[sn].numpy())
            for oname in out_names:
                ref_t = refs[sn][oname]
                onx_t = onx[oname]
                if ref_t.shape[1] != onx_t.shape[1]:
                    print(f"  [NOTE] {vname} {sn} {oname}: NeMo {ref_t.shape[1]} vs ONNX {onx_t.shape[1]} time dim, truncating to {min(ref_t.shape[1], onx_t.shape[1])}")
                min_t = min(ref_t.shape[1], onx_t.shape[1])
                ref_t = ref_t[:, :min_t, :]
                onx_t = onx_t[:, :min_t, :]
                m = metrics(ref_t, onx_t)
                all_rows.append((vname, sn, oname, m, f"{mb:.0f}M"))

    print_table(all_rows)

    # ── 6. Summary (outputs only, averaged across signals) ──
    print("\n\n=== Best-variant summary (outputs only, averaged across signals) ===")
    output_rows = [r for r in all_rows if r[2] == "outputs"]
    variants_seen = set(r[0] for r in output_rows)
    summaries = []
    for v in sorted(variants_seen):
        vrows = [r for r in output_rows if r[0] == v]
        avg_cos = np.mean([r[3]["cos"] for r in vrows]).item()
        avg_mae = np.mean([r[3]["MAE"] for r in vrows]).item()
        sz = vrows[0][4]
        summaries.append((v, avg_cos, avg_mae, sz))
    summaries.sort(key=lambda x: -x[1])
    print(f"{'Rank':>4}  {'Variant':>8}  {'avg cos':>12}  {'avg MAE':>12}  {'Size':>8}  {'Note':>30}")
    print("─" * 75)
    notes = {"F32": "float32 (no quantization)", "A": "bs=32 acc=4 (current)",
             "B": "bs=16 acc=4", "C": "bs=32 acc=5", "D": "bs=16 acc=5",
             "OFF": "official onnx-community"}
    for rank, (v, cos, mae, sz) in enumerate(summaries, 1):
        print(f"{rank:>4}  {v:>8}  {cos:>12.8f}  {mae:>12.6f}  {sz:>8}  {notes.get(v, ''):>30}")

    RelPositionMultiHeadAttention.rel_shift = _orig_rel_shift
    print("\nDone.")


if __name__ == "__main__":
    main()
