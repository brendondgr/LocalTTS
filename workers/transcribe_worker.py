"""One-shot Whisper transcription of a voice reference. Runs under the Breeze venv.

    python transcribe_worker.py reference.wav   ->  {"event": "result", "text": "..."} on the protocol fd

Used when a voice is uploaded without a transcript. The process exits afterwards,
so Whisper never stays resident on the GPU.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import _proto  # noqa: E402

_proto.claim_stdout()

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

MODEL = os.environ.get("LOCALTTS_WHISPER_MODEL", "openai/whisper-large-v3-turbo")


def main() -> None:
    path, language = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        asr = pipeline("automatic-speech-recognition", model=MODEL, dtype=torch.float16, device=device)
        audio, sr = librosa.load(path, sr=16000, mono=True)
        kwargs = {"language": language} if language else {}
        text = asr({"raw": np.ascontiguousarray(audio), "sampling_rate": sr},
                   return_timestamps=True, generate_kwargs=kwargs)["text"].strip()
        _proto.send("result", text=text)
    except Exception as exc:
        _proto.send("error", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
