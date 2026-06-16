#!/usr/bin/env python3
"""
Compare official vs our (dynamo) encoder ONNX outputs numerically.

Usage: uv run python scripts/compare_encoders.py
"""

import os, sys, tempfile, shutil
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

N_MELS = 128
N_LAYERS = 24
D_MODEL = 1024
LEFT_CONTEXT = 56
CONV_CONTEXT = 8

OFFICIAL_REPO = "onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4"
OUR_REPO = "jeffpeng3/nemotron-3.5-asr-streaming-0.6b-onnx-int4-dynamic"


def make_inputs(t=65, lang=101):
    audio = np.random.randn(1, t, N_MELS).astype(np.float32)
    length = np.array([t], dtype=np.int64)
    cache_ch = np.zeros((1, N_LAYERS, LEFT_CONTEXT, D_MODEL), dtype=np.float32)
    cache_tm = np.zeros((1, N_LAYERS, D_MODEL, CONV_CONTEXT), dtype=np.float32)
    cache_len = np.zeros((1,), dtype=np.int64)
    lang_id = np.full((1,), lang, dtype=np.int64)
    return {
        "audio_signal": audio,
        "length": length,
        "cache_last_channel": cache_ch,
        "cache_last_time": cache_tm,
        "cache_last_channel_len": cache_len,
        "lang_id": lang_id,
    }


def download_with_data(repo_id, onnx_filename):
    """Download .onnx and .onnx.data to a temp dir and return the onnx path."""
    tmp = tempfile.mktemp(suffix="_onnx_compare")
    os.makedirs(tmp, exist_ok=True)

    onnx_path = hf_hub_download(repo_id=repo_id, filename=onnx_filename)
    # Symlink into temp dir so ORT finds the .data file alongside
    dest_onnx = os.path.join(tmp, os.path.basename(onnx_path))
    shutil.copy2(onnx_path, dest_onnx)

    data_fn = onnx_filename + ".data"
    try:
        data_path = hf_hub_download(repo_id=repo_id, filename=data_fn)
        dest_data = os.path.join(tmp, data_fn)
        shutil.copy2(data_path, dest_data)
    except Exception:
        pass  # some models embed weights
    return dest_onnx


def load_model(repo_id, filename, providers):
    path = download_with_data(repo_id, filename)
    sess = ort.InferenceSession(path, providers=providers)
    print(f"  {filename}: {sess.get_modelmeta().graph_name}, "
          f"inputs={[i.name for i in sess.get_inputs()]}, "
          f"outputs={[o.name for o in sess.get_outputs()]}")
    return sess, path


def run_and_get(sess, feeds, out_names):
    res = sess.run(out_names, feeds)
    return dict(zip(out_names, res))


def main():
    # Official encoder is static-shaped (T=65 for NORMAL profile).
    # Our dynamo encoder handles all T (17–121).
    prof_name = "NORMAL"
    t = 65

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = ["CPUExecutionProvider"]

    out_names = ["outputs", "cache_last_channel_next", "cache_last_time_next"]

    print("=" * 60)
    print("Encoder ONNX numeric comparison: official vs dynamo")
    print("=" * 60)

    print("\nLoading official encoder (onnx-community) ...")
    off_sess, off_path = load_model(OFFICIAL_REPO, "encoder.onnx", providers)

    print("\nLoading our (dynamo) encoder ...")
    our_sess, our_path = load_model(OUR_REPO, "encoder.onnx", providers)

    print(f"\n── {prof_name} (T={t}) ──")

    feeds = make_inputs(t=t)

    off_out = run_and_get(off_sess, feeds, out_names)
    our_out = run_and_get(our_sess, feeds, out_names)

    for name in out_names:
        a = off_out[name]
        b = our_out[name]
        mae = np.mean(np.abs(a - b))
        mse = np.mean((a - b) ** 2)
        max_err = np.max(np.abs(a - b))
        eps = 1e-8
        cos_sim = np.dot(a.ravel(), b.ravel()) / (
            np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + eps
        )
        print(f"  {name:30s}  MAE={mae:.6f}  MSE={mse:.6f}  "
              f"MAX={max_err:.6f}  cos={cos_sim:.8f}  shape={tuple(a.shape)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
