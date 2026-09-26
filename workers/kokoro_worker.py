"""Kokoro-82M worker. Runs under the Kokoro venv (default ~/venvs/kokoro), read-only.

Built-in voices only (Kokoro cannot clone). The voice prefix picks the language:
a* = American English, b* = British English. One model is shared by both pipelines.
"""

from __future__ import annotations

import os
import time
import warnings

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import _proto  # noqa: E402

_proto.claim_stdout()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from kokoro import KPipeline  # noqa: E402

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000


class Kokoro:
    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipelines = {"a": KPipeline(lang_code="a", repo_id=REPO_ID, device=self.device)}
        self.load_s = time.perf_counter() - t0

    def pipeline(self, lang: str) -> KPipeline:
        if lang not in self.pipelines:
            self.pipelines[lang] = KPipeline(lang_code=lang, repo_id=REPO_ID,
                                             model=self.pipelines["a"].model)
        return self.pipelines[lang]

    def generate(self, text: str, voice: str = "af_heart", speed: float = 1.0,
                 request_id: str = "req", **_ignored) -> None:
        lang = voice[0]
        if lang not in ("a", "b"):
            raise ValueError(f"unsupported Kokoro voice {voice!r} (only a*/b* English voices)")
        t_start = time.perf_counter()
        ttfa, samples = None, 0
        for r in self.pipeline(lang)(text, voice=voice, speed=speed):
            if r.audio is None:
                continue
            audio = r.audio.detach().cpu().numpy().astype(np.float32)
            if ttfa is None:
                ttfa = time.perf_counter() - t_start
            _proto.send_audio(audio)
            samples += len(audio)
        gen_s = time.perf_counter() - t_start
        audio_s = samples / SAMPLE_RATE
        _proto.send("done", audio_s=round(audio_s, 3), gen_s=round(gen_s, 3),
                    rtf=round(gen_s / audio_s, 3) if audio_s else None,
                    ttfa_ms=round(ttfa * 1000, 1) if ttfa is not None else None)


def main() -> None:
    engine = Kokoro()
    _proto.log(f"kokoro ready in {engine.load_s:.1f}s on {engine.device}")
    _proto.send("ready", sample_rate=SAMPLE_RATE, load_s=round(engine.load_s, 1),
                torch=torch.__version__, device=engine.device)
    _proto.serve({"generate": engine.generate})


if __name__ == "__main__":
    main()
