"""Breeze TTS 2 worker. Runs under the Breeze venv (default ~/venvs/breeze-next).

Loads the model once, warms the fast stages, then serves generate requests over the
protocol in _proto.py. Covers voice design (instruction only), voice clone (reference
audio + transcript) and voice direction (both). Long text is split at sentence
boundaries so no single pass runs into the 1500-frame (120 s) generation cap.
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")

import _proto  # noqa: E402

_proto.claim_stdout()

REPO = Path(os.environ.get("BREEZE_REPO", "~/Projects/breeze-tts")).expanduser()
MODEL = Path(os.environ.get("BREEZE_MODEL", "~/models/breeze-tts-2")).expanduser()
FAST = [s for s in os.environ.get("LOCALTTS_BREEZE_FAST", "depth_decoder,backbone_decode").split(",") if s]
STAGES = ("text_encoder", "backbone_prefill", "backbone_decode", "depth_decoder", "codec")
SEGMENT_CHARS = int(os.environ.get("LOCALTTS_BREEZE_SEGMENT_CHARS", "400"))
GAP_S = 0.25

sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from breeze_infer.runtime import (  # noqa: E402
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs, select_template_name  # noqa: E402
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig  # noqa: E402
from models.warmup_profile import load_warmup_profile  # noqa: E402


def split_text(text: str, limit: int = SEGMENT_CHARS) -> list[str]:
    """Pack whole sentences into segments of at most `limit` characters where possible."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?。！？…])\s+", text)
    segments, cur = [], ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > limit:
            segments.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        segments.append(cur)
    return segments


class Breeze:
    def __init__(self) -> None:
        unknown = set(FAST) - set(STAGES)
        if unknown:
            raise ValueError(f"LOCALTTS_BREEZE_FAST has unknown stages: {sorted(unknown)}")
        t0 = time.perf_counter()
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(
            MODEL, device=resolve_device(), attn_implementation="eager"
        )
        update_generation_config_for_breeze(self.model)
        config = FastStreamingConfig(
            max_new_tokens=1500,
            max_seq_len=2048,
            repetition_penalty=1.1,
            **{f"fast_{s}": s in FAST for s in STAGES},
        )
        self.runtime = FastBreezeStreamingRuntime(
            self.model, self.audio_tokenizer, config, tokenizer=self.tokenizer
        )
        if self.runtime.fast_enabled:
            profile = load_warmup_profile(REPO / "configs" / "fast.json")
            profile = replace(profile, codec_chunk_frames=self.runtime.codec_chunk_frames)
            self.runtime.warmup_from_profile(profile)
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.sample_rate = self.runtime.sample_rate

    def generate(self, text: str, instruction: str | None = None, ref_audio: str | None = None,
                 ref_text: str | None = None, cfg_scale: float | None = None, seed: int = 42,
                 request_id: str = "req") -> None:
        instruction = (instruction or "").strip() or None
        if cfg_scale is None:
            cfg_scale = 4.0 if instruction else 1.0  # upstream guidance for design/direction
        t_start = time.perf_counter()
        ttfa, samples = None, 0
        segments = split_text(text)
        for n, segment in enumerate(segments):
            request = {"id": f"{request_id}-{n}", "text": segment, "speaker": "S0"}
            if instruction:
                request["instruction"] = instruction
            if ref_audio:
                request["ref_audio_path"] = ref_audio
                request["ref_text"] = (ref_text or "").strip()
            set_all_seeds(seed)
            inputs = prepare_inputs(
                self.tokenizer, self.audio_tokenizer, self.model, [request],
                get_template(select_template_name(request)),
                guidance_scale=cfg_scale, guidance_scale_ref=None, guidance_scale_ins=None,
            )
            if n:
                gap = np.zeros(int(GAP_S * self.sample_rate), np.float32)
                _proto.send_audio(gap)
                samples += len(gap)
            set_all_seeds(seed)
            for chunk in self.runtime.iter_audio_chunks(inputs, request_id=request["id"], seed=seed):
                audio = np.asarray(chunk.audio, dtype=np.float32)
                if not np.isfinite(audio).all():
                    raise RuntimeError("model produced NaN/inf audio")
                if ttfa is None:
                    ttfa = time.perf_counter() - t_start
                _proto.send_audio(audio)
                samples += len(audio)
        gen_s = time.perf_counter() - t_start
        audio_s = samples / self.sample_rate
        _proto.send("done", audio_s=round(audio_s, 3), gen_s=round(gen_s, 3),
                    rtf=round(gen_s / audio_s, 3) if audio_s else None,
                    ttfa_ms=round(ttfa * 1000, 1) if ttfa is not None else None,
                    segments=len(segments), cfg_scale=cfg_scale)


def main() -> None:
    engine = Breeze()
    _proto.log(f"breeze ready in {engine.load_s:.1f}s, fast stages: {FAST or 'none'}")
    _proto.send("ready", sample_rate=engine.sample_rate, load_s=round(engine.load_s, 1),
                fast=FAST, torch=torch.__version__,
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    _proto.serve({"generate": engine.generate})


if __name__ == "__main__":
    main()
