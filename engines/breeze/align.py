"""Word timings for speech whose model doesn't report them (Breeze), via Whisper.

Whisper transcribes the clip with word timestamps; those are matched to the *script's* words
(difflib), so the timings carry the exact words and punctuation of the script, as kokoro-tts
does. Words Whisper heard differently ("2026" vs "twenty twenty-six") get times interpolated
from their neighbours. Vocal-event tags such as (sigh) are not words and are left out.

Shared by breeze-tts (video-maker) and LocalTTS's /v1/align. Keep the copies identical.
"""
from __future__ import annotations

import difflib
import re

import numpy as np

TAGS = {"laugh", "laughs", "sigh", "sighs", "cough", "coughs", "clears throat", "chuckle", "giggle",
        "gasp", "groan", "sniff", "breath", "hmm", "cry"}
_TAG = re.compile(r"[(\[]\s*([A-Za-z][A-Za-z ]{0,30}?)\s*[)\]]")


def strip_tags(text: str) -> str:
    return re.sub(r"\s{2,}", " ", _TAG.sub(lambda m: " " if m.group(1).lower() in TAGS else m.group(0), text)).strip()


def script_words(text: str) -> list[str]:
    return strip_tags(text).split()


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


class Aligner:
    def __init__(self, model: str = "openai/whisper-large-v3-turbo", device: str = "cuda"):
        import torch
        from transformers import pipeline
        gpu = device.startswith("cuda") and torch.cuda.is_available()
        self.pipe = pipeline("automatic-speech-recognition", model=model,
                             dtype=torch.float16 if gpu else torch.float32, device=device if gpu else "cpu",
                             ignore_warning=True)   # chunked long-form is fine for word timings

    def align(self, audio: np.ndarray, sr: int, text: str, language: str | None = "en") -> dict:
        import librosa
        duration = len(audio) / sr
        y = librosa.resample(np.asarray(audio, np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else audio
        kw = {"return_timestamps": "word", "chunk_length_s": 30}
        if language:
            kw["generate_kwargs"] = {"language": language, "task": "transcribe"}
        heard = []
        for c in self.pipe(y.astype(np.float32), **kw).get("chunks", []):
            start, end = c["timestamp"]
            if start is None:
                continue
            heard.append((c["text"].strip(), float(start), float(end if end is not None else duration)))
        return {"duration": round(duration, 3), "sample_rate": sr, "words": match(script_words(text), heard, duration)}


def match(words: list[str], heard: list[tuple[str, float, float]], duration: float) -> list[dict]:
    """Give each script word a (start, end), using Whisper's words where they line up."""
    times: list[tuple[float, float] | None] = [None] * len(words)
    a, b = [_norm(w) for w in words], [_norm(h[0]) for h in heard]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = heard[j1 + k][1:]
        elif op == "replace" and j2 > j1:
            # Same stretch, worded differently: share the heard span out by character count.
            t0, t1 = heard[j1][1], heard[j2 - 1][2]
            _spread(words, times, i1, i2, t0, t1)
    # Anything still unplaced sits between its placed neighbours.
    i = 0
    while i < len(words):
        if times[i] is None:
            j = i
            while j < len(words) and times[j] is None:
                j += 1
            t0 = times[i - 1][1] if i else min(0.1, duration)
            t1 = times[j][0] if j < len(words) else max(t0, duration - 0.1)
            _spread(words, times, i, j, t0, max(t0, t1))
            i = j
        else:
            i += 1
    out, last = [], 0.0
    for k, (w, (s, e)) in enumerate(zip(words, times)):
        # Whisper stretches a word over neighbouring non-speech (a sigh before the first word,
        # a pause after a word). Cap a word's length: trim the front of the first word, the tail of others.
        cap = 0.35 + 0.1 * len(_norm(w))
        if e - s > cap:
            s, e = (e - cap, e) if k == 0 else (s, s + cap)
        s = max(s, last)
        e = max(e, s + 0.01)
        out.append({"w": w, "start": round(s, 3), "end": round(min(e, duration), 3)})
        last = s
    return out


def _spread(words, times, i1, i2, t0, t1):
    lens = [max(1, len(_norm(w))) for w in words[i1:i2]]
    total, t = sum(lens), t0
    for k, n in enumerate(lens):
        d = (t1 - t0) * n / total
        times[i1 + k] = (t, t + d)
        t += d
