export const Profiles = {
  TURBO:    { encoder: "encoder_80ms.onnx",   encoderData: "encoder_80ms.onnx.data",   latencyMs: 80,  newFrames: 8  },
  FAST:     { encoder: "encoder_160ms.onnx",  encoderData: "encoder_160ms.onnx.data",  latencyMs: 160, newFrames: 16 },
  BALANCED: { encoder: "encoder_320ms.onnx",  encoderData: "encoder_320ms.onnx.data",  latencyMs: 320, newFrames: 32 },
  NORMAL:   { encoder: "encoder_560ms.onnx",  encoderData: "encoder_560ms.onnx.data",  latencyMs: 560, newFrames: 56 },
  HIGH:     { encoder: "encoder_1120ms.onnx", encoderData: "encoder_1120ms.onnx.data", latencyMs: 1120, newFrames: 112 },
};

export const CONFIG = {
  BASE: "https://huggingface.co/jeffpeng3/nemotron-3.5-asr-streaming-0.6b-onnx-int4/resolve/main/",
  SR: 16000,
  N_FFT: 512,
  HOP: 160,
  WIN: 400,
  N_MELS: 128,
  FMIN: 0,
  FMAX: 8000,
  PREEMPH: 0.97,
  LOG_GUARD: 1e-10,
  NEW_FRAMES: 56,
  CACHE_FRAMES: 9,
  LAYERS: 24,
  D_MODEL: 1024,
  DEC_HID: 640,
  DEC_LAYERS: 2,
  VOCAB: 13088,
  BLANK: 13087,
  MAX_SYM: 10,
};


