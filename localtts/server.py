"""LocalTTS HTTP API. See README.md for the endpoint reference and curl examples."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

from . import __version__, audio, voices
from .config import settings
from .engines import ENGINES, EngineError, transcribe

log = logging.getLogger("localtts")

MAX_TEXT = 20_000
MIN_REF_S, MAX_REF_S, WARN_REF_S = 3.0, 60.0, 30.0
DESIGN_SAMPLE = ("Hello, and welcome. This is a short sample of my voice, recorded so that "
                 "I sound the same every time you hear me.")
Format = Literal["wav", "flac", "mp3", "ogg", "pcm"]


# ---- app & lifecycle ---------------------------------------------------------------------

async def _reaper() -> None:
    while True:
        await asyncio.sleep(15)
        for engine in ENGINES.values():
            try:
                await engine.maybe_idle_unload()
            except Exception:
                log.exception("idle unload failed for %s", engine.name)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    reaper = asyncio.create_task(_reaper())
    for name in settings.preload:
        if name in ENGINES:
            asyncio.create_task(ENGINES[name].ensure_loaded())
    yield
    reaper.cancel()
    for engine in ENGINES.values():
        await engine.unload(reason="server shutting down")


app = FastAPI(title="LocalTTS", version=__version__, lifespan=lifespan)


@app.exception_handler(EngineError)
async def _engine_error(_, exc: EngineError):
    return JSONResponse({"detail": str(exc)}, status_code=500)


def _engine(name: str):
    if name not in ENGINES:
        raise HTTPException(400, f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    return ENGINES[name]


@app.get("/")
def root():
    return {"service": "localtts", "version": __version__, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "idle_unload_s": settings.idle_unload_s,
        "engines": {n: e.status() for n, e in ENGINES.items()},
        "voices": len(voices.all_voices()),
        "data": str(settings.data),
    }


@app.post("/v1/load")
async def load(engine: str = Query("breeze")):
    e = _engine(engine)
    await e.ensure_loaded()
    e.last_used = time.monotonic()
    return {"engine": engine, **e.status()}


@app.post("/v1/unload")
async def unload(engine: str | None = Query(None, description="omit to unload all")):
    targets = [_engine(engine)] if engine else list(ENGINES.values())
    return {"unloaded": [e.name for e in targets if await e.unload()]}


# ---- voices ------------------------------------------------------------------------------

def _voice_or_404(name: str) -> voices.Voice:
    v = voices.get(name)
    if v is None:
        raise HTTPException(404, f"no voice named {name!r}")
    return v


@app.get("/v1/voices")
def list_voices():
    return {
        "custom": [v.public() for v in voices.all_voices()],
        "kokoro": voices.KOKORO_VOICES,
        "breeze_design": "no name + an `instruction` describing the voice (or save one with POST /v1/voices/design)",
    }


@app.get("/v1/voices/{name}")
def voice_info(name: str):
    return _voice_or_404(name).public()


@app.get("/v1/voices/{name}/audio")
def voice_audio(name: str):
    v = _voice_or_404(name)
    return FileResponse(v.reference, media_type="audio/wav", filename=f"{voices.slug(v.name)}.wav")


def _claim_name(name: str, overwrite: bool) -> str:
    try:
        name = voices.check_name(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if voices.get(name) is not None and not overwrite:
        raise HTTPException(409, f"voice {name!r} exists; pass overwrite=true to replace it")
    return name


@app.post("/v1/voices", status_code=201)
async def add_voice(
    name: str = Form(..., description="what you will call it with, e.g. name=Brendon"),
    audio_file: UploadFile = File(..., alias="audio", description="reference clip, any format ffmpeg reads"),
    transcript: str = Form("", description="exact words spoken in the clip; omit to auto-transcribe"),
    language: str | None = Form(None, description="Whisper language hint, e.g. en or zh"),
    overwrite: bool = Form(False),
):
    name = _claim_name(name, overwrite)
    payload = await audio_file.read()
    if not payload:
        raise HTTPException(400, "the audio upload is empty")
    # Staged next to the voices (same filesystem) so the final move is an atomic rename.
    with tempfile.TemporaryDirectory(prefix=".upload-", dir=settings.voices_dir) as tmp:
        raw = Path(tmp) / ("upload" + (Path(audio_file.filename or "").suffix or ".bin"))
        raw.write_bytes(payload)
        ref = Path(tmp) / "reference.wav"
        try:
            await audio.to_reference_wav(str(raw), str(ref))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        dur = audio.duration_s(str(ref))
        if not MIN_REF_S <= dur <= MAX_REF_S:
            raise HTTPException(400, f"reference is {dur:.1f}s; use {MIN_REF_S:.0f}-{MAX_REF_S:.0f}s "
                                     f"(10-25s of clean speech works best)")
        transcript = transcript.strip()
        auto = not transcript
        if auto:
            transcript = await transcribe(ref, language)
            if not transcript:
                raise HTTPException(400, "Whisper heard no speech in the clip")
        voice = voices.Voice(name=name, transcript=transcript, duration_s=round(dur, 2),
                             created=voices.now(), language=language, transcript_auto=auto)
        voices.delete(name)  # overwrite: drop the old folder (permanently) before writing the new one
        voice.dir.mkdir(parents=True)
        try:
            ref.replace(voice.reference)
            voices.save(voice)
        except Exception:
            voices.delete(name)
            raise
    out = voice.public()
    if dur > WARN_REF_S:
        out["warning"] = f"{dur:.0f}s is long; 10-25s references clone just as well and generate faster"
    if auto:
        out["note"] = "transcript written by Whisper; check it, since clone quality depends on it " \
                      "(fix with PATCH /v1/voices/{name})"
    return out


class VoicePatch(BaseModel):
    transcript: str | None = None
    rename: str | None = None


@app.patch("/v1/voices/{name}")
def edit_voice(name: str, body: VoicePatch):
    v = _voice_or_404(name)
    if body.transcript is not None:
        if not body.transcript.strip():
            raise HTTPException(400, "transcript cannot be empty")
        v.transcript, v.transcript_auto = body.transcript.strip(), False
    if body.rename and voices.slug(body.rename) != voices.slug(v.name):
        new = _claim_name(body.rename, overwrite=False)
        v.dir.rename(voices.voice_dir(new))
        v.name = new
    elif body.rename:
        v.name = body.rename.strip()
    voices.save(v)
    return v.public()


@app.delete("/v1/voices/{name}")
def delete_voice(name: str):
    v = _voice_or_404(name)
    voices.delete(v.name)
    return {"deleted": v.name, "permanent": True}


class DesignRequest(BaseModel):
    name: str
    instruction: str = Field(..., description="natural-language description of the voice")
    sample_text: str = DESIGN_SAMPLE
    seed: int = 42
    cfg_scale: float = 4.0
    overwrite: bool = False


@app.post("/v1/voices/design", status_code=201)
async def design_voice(body: DesignRequest):
    """Invent a voice from a description, then keep it as a named voice so it stays consistent."""
    name = _claim_name(body.name, body.overwrite)
    pcm, stats = await _collect(ENGINES["breeze"], dict(
        text=body.sample_text, instruction=body.instruction, cfg_scale=body.cfg_scale, seed=body.seed))
    voices.delete(name)
    voice = voices.Voice(name=name, transcript=body.sample_text, duration_s=round(stats["audio_s"], 2),
                         created=voices.now(), source="design", instruction=body.instruction)
    voice.dir.mkdir(parents=True)
    voice.reference.write_bytes(audio.encode(pcm, ENGINES["breeze"].info.get("sample_rate", 24000), "wav"))
    voices.save(voice)
    return {**voice.public(), "stats": stats}


# ---- speech ------------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    text: str = Field(..., validation_alias=AliasChoices("text", "input"))
    name: str | None = Field(None, validation_alias=AliasChoices("name", "voice"),
                             description="custom voice name, a Kokoro voice (af_heart...), or omit")
    engine: Literal["auto", "breeze", "kokoro"] = "auto"
    instruction: str | None = Field(None, validation_alias=AliasChoices("instruction", "instructions"),
                                    description="Breeze: describe the voice (design) or the delivery (direction)")
    cfg_scale: float | None = Field(None, gt=0, le=10, description="Breeze guidance; default 1, or 4 with an instruction")
    seed: int = 42
    speed: float = Field(1.0, gt=0.25, le=4.0, description="Kokoro only")
    format: Format = Field("wav", validation_alias=AliasChoices("format", "response_format"))
    stream: bool = Field(False, description="stream raw PCM s16le as it is generated")
    no_save: bool = Field(False, description="return the audio only; nothing is written to disk")


def _resolve(req: SpeechRequest) -> tuple[str, str, dict]:
    """-> (engine, voice label, worker params)"""
    if len(req.text) > MAX_TEXT:
        raise HTTPException(413, f"text is over {MAX_TEXT} characters")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    name = (req.name or "").strip() or None
    if name and name.lower() in voices.KOKORO_VOICES:
        if req.engine == "breeze":
            raise HTTPException(400, f"{name!r} is a Kokoro voice; Breeze uses custom voices or an instruction")
        if req.instruction:
            raise HTTPException(400, "Kokoro does not take an instruction")
        return "kokoro", name.lower(), dict(text=req.text, voice=name.lower(), speed=req.speed)
    if name:
        v = voices.get(name)
        if v is None:
            raise HTTPException(404, f"no voice named {name!r}; see GET /v1/voices")
        if req.engine == "kokoro":
            raise HTTPException(400, "custom voices are Breeze voices; Kokoro cannot clone")
        return "breeze", v.name, dict(text=req.text, instruction=req.instruction, ref_audio=str(v.reference),
                                      ref_text=v.transcript, cfg_scale=req.cfg_scale, seed=req.seed)
    engine = req.engine if req.engine != "auto" else ("breeze" if req.instruction else settings.default_engine)
    if engine == "kokoro":
        if req.instruction:
            raise HTTPException(400, "Kokoro does not take an instruction")
        v = settings.default_kokoro_voice
        return "kokoro", v, dict(text=req.text, voice=v, speed=req.speed)
    return "breeze", "design", dict(text=req.text, instruction=req.instruction,
                                    cfg_scale=req.cfg_scale, seed=req.seed)


async def _collect(engine, params: dict) -> tuple[bytes, dict]:
    parts, stats = [], {}
    gen = engine.generate(request_id=uuid.uuid4().hex[:8], **params)
    try:
        async for item in gen:
            if isinstance(item, dict):
                stats = {k: v for k, v in item.items() if k != "event"}
            else:
                parts.append(item)
    finally:
        await gen.aclose()
    return b"".join(parts), stats


def _output_path(voice: str, fmt: str) -> Path:
    stem = f"{time.strftime('%Y%m%d-%H%M%S')}_{voices.slug(voice) or 'voice'}_{uuid.uuid4().hex[:6]}"
    return settings.outputs_dir / f"{stem}.{fmt if fmt != 'pcm' else 'wav'}"


async def _speak(req: SpeechRequest):
    engine_name, voice, params = _resolve(req)
    engine = ENGINES[engine_name]
    base_headers = {"X-LocalTTS-Engine": engine_name, "X-LocalTTS-Voice": voice, "Cache-Control": "no-store"}

    if req.stream:
        await engine.ensure_loaded()
        sr = engine.info.get("sample_rate", 24000)
        out_path = None if req.no_save else _output_path(voice, "wav")

        async def body():
            parts = []
            gen = engine.generate(request_id=uuid.uuid4().hex[:8], **params)
            try:
                async for item in gen:
                    if isinstance(item, bytes):
                        if out_path:
                            parts.append(item)
                        yield item
            finally:
                await gen.aclose()
            if out_path:
                out_path.write_bytes(audio.encode(b"".join(parts), sr, "wav"))

        headers = {**base_headers, "X-Sample-Rate": str(sr), "X-Sample-Format": "s16le", "X-Channels": "1",
                   "X-LocalTTS-Saved": out_path.name if out_path else "no"}
        return StreamingResponse(body(), media_type="audio/pcm", headers=headers)

    pcm, stats = await _collect(engine, params)
    sr = engine.info.get("sample_rate", 24000)
    data = audio.encode(pcm, sr, req.format)
    headers = {**base_headers, "X-Sample-Rate": str(sr),
               "X-Audio-Duration": str(stats.get("audio_s")), "X-RTF": str(stats.get("rtf")),
               "X-Generation-Seconds": str(stats.get("gen_s"))}
    if req.no_save:
        headers["X-LocalTTS-Saved"] = "no"
    else:
        out = _output_path(voice, req.format)
        out.write_bytes(data if req.format != "pcm" else audio.encode(pcm, sr, "wav"))
        headers["X-LocalTTS-Saved"] = out.name
        headers["X-LocalTTS-Path"] = str(out)
    if req.format == "pcm":
        headers["X-Sample-Format"] = "s16le"
    ext = req.format
    headers["Content-Disposition"] = f'inline; filename="{voices.slug(voice) or "speech"}.{ext}"'
    return Response(data, media_type=audio.MEDIA_TYPES[req.format], headers=headers)


@app.post("/v1/speech")
async def speech(req: SpeechRequest):
    return await _speak(req)


class OpenAISpeech(BaseModel):
    """OpenAI-style body, so tools that speak /v1/audio/speech work unchanged."""
    model: str = "auto"
    input: str
    voice: str | None = None
    instructions: str | None = None
    response_format: Format = "mp3"
    speed: float = 1.0
    no_save: bool = False


@app.post("/v1/audio/speech")
async def openai_speech(body: OpenAISpeech):
    engine = body.model if body.model in ("breeze", "kokoro") else "auto"
    req = SpeechRequest(text=body.input, name=body.voice, engine=engine, instruction=body.instructions,
                        format=body.response_format, speed=body.speed, no_save=body.no_save)
    return await _speak(req)


# ---- saved outputs -----------------------------------------------------------------------

_SAFE_FILE = re.compile(r"^[A-Za-z0-9._\-]+$")


def _output_or_404(filename: str) -> Path:
    p = settings.outputs_dir / filename
    if not _SAFE_FILE.match(filename) or not p.is_file():
        raise HTTPException(404, f"no saved output {filename!r}")
    return p


@app.get("/v1/outputs")
def list_outputs():
    files = sorted(settings.outputs_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"dir": str(settings.outputs_dir), "files": [
        {"file": p.name, "bytes": p.stat().st_size,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))}
        for p in files if p.is_file()]}


@app.get("/v1/outputs/{filename}")
def get_output(filename: str):
    return FileResponse(_output_or_404(filename))


@app.delete("/v1/outputs/{filename}")
def delete_output(filename: str):
    _output_or_404(filename).unlink()  # permanent, no trash
    return {"deleted": filename, "permanent": True}
