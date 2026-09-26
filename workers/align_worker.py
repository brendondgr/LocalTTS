"""One-shot word alignment for finished clips. Runs under the Breeze venv (it has Whisper).

    python align_worker.py job.json   job = [{"id", "text", "path"}]
    ->  {"event": "result", "items": {id: {"duration", "sample_rate", "words": [{w, start, end}]}}}

Used by POST /v1/align (video-maker's localtts provider asks for timings after synthesis).
Whisper loads once for the whole batch, and the process exits afterwards.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import _proto  # noqa: E402

_proto.claim_stdout()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from align import Aligner  # noqa: E402

MODEL = os.environ.get("LOCALTTS_WHISPER_MODEL", "openai/whisper-large-v3-turbo")


def main() -> None:
    try:
        job = json.loads(open(sys.argv[1]).read())
        aligner = Aligner(MODEL, "cuda" if torch.cuda.is_available() else "cpu")
        out = {}
        for item in job:
            audio, sr = sf.read(item["path"], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            out[item["id"]] = aligner.align(audio, sr, item["text"], item.get("language", "en"))
        _proto.send("result", items=out)
    except Exception as exc:
        _proto.send("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
