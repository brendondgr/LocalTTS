"""Audio helpers: PCM -> container bytes, and normalising uploaded references with ffmpeg."""

from __future__ import annotations

import asyncio
import io
import shutil

import numpy as np
import soundfile as sf

MEDIA_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "pcm": "audio/pcm",
}
SF_FORMATS = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"),
              "mp3": ("MP3", "MPEG_LAYER_III"), "ogg": ("OGG", "VORBIS")}


def encode(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return pcm
    audio = np.frombuffer(pcm, dtype="<i2")
    buf = io.BytesIO()
    container, subtype = SF_FORMATS[fmt]
    # libsndfile's MP3 default is a very low VBR bitrate; 0.0 = best (160 kbps CBR at 24 kHz).
    extra = {"compression_level": 0.0, "bitrate_mode": "CONSTANT"} if fmt == "mp3" else {}
    sf.write(buf, audio, sample_rate, format=container, subtype=subtype, **extra)
    return buf.getvalue()


async def to_reference_wav(src: str, dst: str, sample_rate: int = 24_000) -> None:
    """Any audio ffmpeg can read -> mono 16-bit WAV at the model rate."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
        "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16", dst,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise ValueError(f"could not decode the audio: {err.decode().strip()[-400:]}")


def duration_s(path: str) -> float:
    return sf.info(path).duration
