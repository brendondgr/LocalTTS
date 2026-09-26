"""Named voices: a reference clip plus its exact transcript, used by Breeze voice cloning.

    data/voices/<slug>/reference.wav   mono 16-bit 24 kHz
    data/voices/<slug>/voice.json      {"name", "transcript", "duration_s", "created", "source", ...}

Deleting a voice removes its folder outright (unlink/rmtree, never a trash can).
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import settings

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}$")

# Kokoro's built-in English voices (a* = American, b* = British).
KOKORO_VOICES = (
    "af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
    "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa "
    "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis"
).split()


@dataclass
class Voice:
    name: str
    transcript: str
    duration_s: float
    created: str
    source: str = "upload"          # upload | design
    instruction: str | None = None  # the design prompt, for designed voices
    language: str | None = None
    transcript_auto: bool = False   # True when Whisper wrote the transcript

    @property
    def dir(self) -> Path:
        return voice_dir(self.name)

    @property
    def reference(self) -> Path:
        return self.dir / "reference.wav"

    def public(self) -> dict:
        return {"engine": "breeze", **asdict(self)}


def slug(name: str) -> str:
    return re.sub(r"[ _.]+", "-", name.strip().lower()).strip("-")


def check_name(name: str) -> str:
    name = name.strip()
    if not NAME_RE.match(name):
        raise ValueError("voice names are 1-64 characters: letters, digits, space, _ . -")
    if name.lower() in KOKORO_VOICES:
        raise ValueError(f"{name!r} is a built-in Kokoro voice name")
    return name


def voice_dir(name: str) -> Path:
    return settings.voices_dir / slug(name)


def get(name: str) -> Voice | None:
    meta = voice_dir(name) / "voice.json"
    if not meta.is_file():
        return None
    return Voice(**json.loads(meta.read_text()))


def all_voices() -> list[Voice]:
    if not settings.voices_dir.is_dir():
        return []
    out = []
    for meta in sorted(settings.voices_dir.glob("[!.]*/voice.json")):
        try:
            out.append(Voice(**json.loads(meta.read_text())))
        except Exception:
            continue
    return out


def save(voice: Voice) -> None:
    voice.dir.mkdir(parents=True, exist_ok=True)
    tmp = voice.dir / "voice.json.tmp"
    tmp.write_text(json.dumps(asdict(voice), indent=2, ensure_ascii=False))
    tmp.replace(voice.dir / "voice.json")


def delete(name: str) -> bool:
    d = voice_dir(name)
    if not d.is_dir():
        return False
    shutil.rmtree(d)  # permanent
    return True


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
