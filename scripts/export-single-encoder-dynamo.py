#!/usr/bin/env python3
"""
Export single encoder (rc=13) with dynamo=True / opset=21.
Static shapes → add dynamic axes post-export.

"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nemo.collections.asr as nemo_asr
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import Dim

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b"
N_MELS = 128
SUBSAMPLING = 8
N_LAYERS = 24
D_MODEL = 1024
CONV_CONTEXT = 8
PRE_ENCODE_CACHE = 9
NUM_PROMPTS = 128
LEFT_CONTEXT = 56
RIGHT_CONTEXT = 13
NEW_FRAMES = 112  # max profile = 1120ms
ENC_IN = NEW_FRAMES + PRE_ENCODE_CACHE

ENCODER_INPUT_NAMES = [
    "audio_signal", "length",
    "cache_last_channel", "cache_last_time", "cache_last_channel_len",
    "lang_id",
]
ENCODER_OUTPUT_NAMES = [
    "outputs", "encoded_lengths",
    "cache_last_channel_next", "cache_last_time_next",
    "cache_last_channel_len_next",
]


class StreamingEncoderWrapper(nn.Module):
    def __init__(self, encoder, prompt_kernel):
        super().__init__()
        self.encoder = encoder
        self.prompt_kernel = prompt_kernel

    def forward(self, audio_signal, length,
                cache_last_channel, cache_last_time, cache_last_channel_len,
                lang_id):
        audio_signal = audio_signal.transpose(1, 2)
        encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next = \
            self.encoder.forward_for_export(
                audio_signal=audio_signal,
                length=length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
            )
        encoded = encoded.transpose(1, 2)
        onehot = F.one_hot(lang_id, num_classes=NUM_PROMPTS).to(encoded.dtype)
        prompt = onehot.unsqueeze(1).expand(-1, encoded.shape[1], -1)
        concat = torch.cat([encoded, prompt], dim=-1)
        encoded = self.prompt_kernel(concat).to(encoded.dtype)
        return encoded, encoded_len, cache_ch_next, cache_tm_next, cache_len_next


def load_model():
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(MODEL_NAME)
    nemo_files = [f for f in files if f.endswith(".nemo")]
    nemo_path = hf_hub_download(repo_id=MODEL_NAME, filename=nemo_files[0])
    print(f"  Loading from: {nemo_path}")
    t0 = time.time()
    asr_model = nemo_asr.models.ASRModel.restore_from(nemo_path)
    asr_model = asr_model.cpu()
    asr_model.eval()
    print(f"  Loaded in {time.time()-t0:.0f}s")
    return asr_model


def quantize_int4(onnx_path):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    print(f"  Quantizing to INT4 ...")
    model = onnx.load(str(onnx_path), load_external_data=True)
    for init in model.graph.initializer:
        if len(init.external_data) > 0 or init.data_location != 0:
            init.ClearField('external_data')
            init.ClearField('data_location')

    q = MatMulNBitsQuantizer(
        model=model,
        block_size=32,
        is_symmetric=True,
        accuracy_level=4,
    )
    q.process()

    old_data = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if old_data.exists():
        old_data.unlink()
    onnx_path.unlink()

    q.model.save_model_to_file(
        str(onnx_path),
        use_external_data_format=True,
    )
    total_mb = (onnx_path.stat().st_size + old_data.stat().st_size) / (1024 * 1024)
    print(f"  [OK] INT4: ({total_mb:.1f} MB)")


def main():
    output_dir = Path("build/onnx_models-dynamo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(" dynamo=True / opset=21 single encoder (rc=13)")
    print("=" * 55)

    # ── Patch rel_shift: view → reshape ──
    from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttention
    orig_rel_shift = RelPositionMultiHeadAttention.rel_shift
    def patched_rel_shift(self, x):
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.reshape(b, h, -1, qlen)
        x = x[:, :, 1:].reshape(b, h, qlen, pos_len)
        return x
    RelPositionMultiHeadAttention.rel_shift = patched_rel_shift

    # ── Load model ──
    asr_model = load_model()
    asr_model.encoder.set_default_att_context_size([LEFT_CONTEXT, RIGHT_CONTEXT])
    asr_model.set_export_config({"cache_support": True})

    wrapper = StreamingEncoderWrapper(asr_model.encoder, asr_model.prompt_kernel)
    wrapper.eval()

    dummy = (
        torch.randn(1, ENC_IN, N_MELS),
        torch.tensor([ENC_IN], dtype=torch.int64),
        torch.zeros(1, N_LAYERS, LEFT_CONTEXT, D_MODEL),
        torch.zeros(1, N_LAYERS, D_MODEL, CONV_CONTEXT),
        torch.zeros(1, dtype=torch.int64),
        torch.full((1,), 101, dtype=torch.int64),
    )

    output_path = output_dir / "encoder.onnx"
    output_path_data = output_path.with_suffix(output_path.suffix + ".data")

    print(f"\nExporting with dynamo=True / opset=21 (dynamic time dim)...")
    t0 = time.time()
    try:
        result = torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            input_names=ENCODER_INPUT_NAMES,
            output_names=ENCODER_OUTPUT_NAMES,
            opset_version=21,
            dynamo=True,
            dynamic_shapes={
                "audio_signal": {1: Dim("time", min=17, max=121)},
                "length": None,
                "cache_last_channel": None,
                "cache_last_time": None,
                "cache_last_channel_len": None,
                "lang_id": None,
            },
        )
        elapsed = time.time() - t0
        print(f"  [OK] Export succeeded in {elapsed:.1f}s")

        # Run ONNXProgram optimization
        if hasattr(result, "optimize"):
            print("  Running optimize() ...")
            result.optimize()
        if hasattr(result, "save"):
            result.save(str(output_path))

        # If result.save used external data, re-save consolidated
        if not output_path.exists():
            if hasattr(result, "exported_program"):
                from torch.onnx._internal.exporter._core import ONNXProgram
                onnx_program = ONNXProgram(result.exported_program, {})
                onnx_program.save(str(output_path))
            elif hasattr(result, "model_proto"):
                onnx.save_model(
                    result.model_proto,
                    str(output_path),
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=output_path.name + ".data",
                )
        else:
            # Already saved, just add metadata
            m = onnx.load(str(output_path), load_external_data=True)
            for k, v in {"profile": "single_rc13", "left_context": str(LEFT_CONTEXT),
                          "right_context": str(RIGHT_CONTEXT)}.items():
                meta = m.metadata_props.add()
                meta.key = k
                meta.value = v
            onnx.save_model(
                m,
                str(output_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=output_path.name + ".data",
            )

        # Analyze
        m = onnx.load(str(output_path), load_external_data=False)
        print(f"  Nodes: {len(m.graph.node)}")
        opset_info = {(d.domain or 'ai.onnx', d.version) for d in m.opset_import}
        print(f"  Opsets: {sorted(opset_info)}")

        total_mb = (output_path.stat().st_size + output_path_data.stat().st_size) / (1024 * 1024)
        print(f"  [OK] encoder.onnx + data ({total_mb:.1f} MB)")

        # Integrate decoder + joint from existing export
        src = Path("build/onnx_models-single")
        if src.exists():
            import shutil
            for fname in ["decoder.onnx", "decoder.onnx.data",
                           "joint.onnx", "joint.onnx.data",
                           "genai_config.json", "audio_processor_config.json"]:
                sf = src / fname
                if sf.exists():
                    shutil.copy2(sf, output_dir / fname)
                    print(f"  Copied {fname}")
            # Fix genai_config
            cfg = output_dir / "genai_config.json"
            if cfg.exists():
                with open(cfg) as f:
                    c = json.load(f)
                c["model"]["encoder"]["filename"] = "encoder.onnx"
                with open(cfg, "w") as f:
                    json.dump(c, f, indent=2)

        # INT4 quant
        quantize_int4(output_path)

        # Summary
        print(f"\n{'=' * 55}")
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                sz = f.stat().st_size / (1024 * 1024)
                print(f"  {f.name:40s} {sz:>8.1f} MB")
        print(f"{'=' * 55}")

    except Exception as e:
        print(f"\n  [FAIL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        RelPositionMultiHeadAttention.rel_shift = orig_rel_shift

    del wrapper, asr_model
    gc.collect()


if __name__ == "__main__":
    main()
