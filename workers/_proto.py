"""Wire protocol between the LocalTTS server and an engine worker process.

Each worker runs in its engine's own venv and talks over stdin/stdout:

  server -> worker   one JSON object per line: {"op": "generate" | "transcribe" | "ping", ...}
  worker -> server   one JSON object per line, where a {"event": "chunk", "nbytes": N}
                     line is followed by exactly N bytes of 16-bit little-endian mono PCM.

Events: ready (once, after the model loads), chunk, done, result, error.
Model libraries print to stdout freely, so the worker moves the real stdout to a
private file descriptor at startup and points fd 1 at stderr (the service log).
Nothing ever touches disk on the audio path.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

_out = None


def claim_stdout() -> None:
    """Keep the protocol channel for ourselves; everything else printed goes to stderr."""
    global _out
    fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    _out = os.fdopen(fd, "wb", buffering=0)


def send(event: str, **fields) -> None:
    _out.write((json.dumps({"event": event, **fields}) + "\n").encode())


def send_audio(audio) -> None:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    if pcm:
        send("chunk", nbytes=len(pcm))
        _out.write(pcm)


def log(*parts) -> None:
    print("[worker]", *parts, file=sys.stderr, flush=True)


def serve(handlers: dict) -> None:
    """Answer requests until stdin closes. One request at a time."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.pop("op")
            if op == "ping":
                send("result", ok=True)
                continue
            handlers[op](**req)
        except Exception as exc:  # report and keep serving; the server decides what to do
            traceback.print_exc()
            send("error", message=f"{type(exc).__name__}: {exc}")
