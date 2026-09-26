#!/usr/bin/env python3
"""breeze-tts: Breeze TTS 2 (voice clone / design / direction) with word timings.

Installed into its own environment by tts/setup.sh and run through the `breeze-tts` launcher.
The same batch interface as kokoro-tts, so video-maker can use either.

  breeze-tts "Hello there." --voice NAME -o hello.wav            # clone a saved voice
  breeze-tts "Hello." --ref me.wav --ref-text "exact words" -o hello.wav
  breeze-tts "Hello." --instruction "a calm, deep narrator" -o hello.wav
  breeze-tts --batch job.json                                      # used by video-maker
  breeze-tts --voices                                              # saved voices
  breeze-tts --add-voice NAME clip.wav --ref-text "exact words"    # save a voice
  breeze-tts --check                                               # device report + timed test

Batch job:
  { "voice": "Brendon" | null, "instruction": "…" | null, "seed": 42, "cfg_scale": null,
    "items": [ { "id": "hook", "text": "…", "out": "audio/hook.wav", "timings": "audio/hook.json" } ] }
  voice alone = clone; voice + instruction = direction (tone/emotion/pace);
  instruction alone = design. A designed voice is generated once from the instruction and
  every clip is cloned from that sample, so all clips share one voice.
Timings files match kokoro-tts: { "duration", "sample_rate", "words": [ { "w", "start", "end" } ] }.

Licence: the Breeze TTS 2 weights and everything they generate are for research and
non-commercial use only (BreezeBlue licence).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # weights and Whisper are fetched at install time
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
HOME = Path.home()
REGISTRY = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")) / "tts-engines" / "engines.json"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", HOME / ".local" / "share")) / "tts-engines"
STAGES = ("text_encoder", "backbone_prefill", "backbone_decode", "depth_decoder", "codec")
SEGMENT_CHARS = 400
GAP_S = 0.25
DESIGN_SAMPLE = ("Hello, and welcome. This is a short sample of my voice, recorded so that "
                 "I sound the same every time you hear me.")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def config() -> dict:
    try:
        cfg = json.loads(REGISTRY.read_text()).get("engines", {}).get("breeze", {})
        voices = json.loads(REGISTRY.read_text()).get("voices_dirs", [])
    except (OSError, ValueError):
        cfg, voices = {}, []
    env = os.environ.get
    fast = env("BREEZE_FAST")
    return {
        "repo": Path(env("BREEZE_REPO") or cfg.get("repo") or DATA_DIR / "breeze-tts").expanduser(),
        "weights": Path(env("BREEZE_MODEL") or cfg.get("weights") or DATA_DIR / "breeze-tts-2").expanduser(),
        "device": env("BREEZE_DEVICE") or cfg.get("device") or "auto",
        "dtype": env("BREEZE_DTYPE") or cfg.get("dtype") or "bf16",
        "fast": [s for s in (fast.split(",") if fast is not None else cfg.get("fast", [])) if s],
        "whisper": env("BREEZE_WHISPER") or cfg.get("whisper") or "openai/whisper-large-v3-turbo",
        "voices_dirs": [DATA_DIR / "voices"] + [Path(d) for d in voices],
    }


# ---------------------------------------------------------------------------------- voices

def slug(name: str) -> str:
    return re.sub(r"[ _.]+", "-", name.strip().lower()).strip("-")


def find_voice(cfg: dict, name: str) -> tuple[Path, str]:
    """A saved voice (LocalTTS-compatible layout: <dir>/<slug>/{reference.wav, voice.json})."""
    for d in cfg["voices_dirs"]:
        meta = Path(d) / slug(name) / "voice.json"
        if meta.is_file():
            v = json.loads(meta.read_text())
            return meta.parent / "reference.wav", v["transcript"]
    raise SystemExit(f"breeze-tts: no saved voice {name!r}. See `breeze-tts --voices`, or add one with "
                     f"`breeze-tts --add-voice {name} clip.wav --ref-text \"exact words\"`")


def list_voices(cfg: dict) -> list[dict]:
    out, seen = [], set()
    for d in cfg["voices_dirs"]:
        for meta in sorted(Path(d).glob("[!.]*/voice.json")):
            try:
                v = json.loads(meta.read_text())
            except ValueError:
                continue
            if slug(v["name"]) not in seen:
                seen.add(slug(v["name"]))
                out.append({"name": v["name"], "duration_s": v.get("duration_s"), "dir": str(meta.parent)})
    return out


# ---------------------------------------------------------------------------------- engine

def split_text(text: str, limit: int = SEGMENT_CHARS) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    segments, cur = [], ""
    for s in re.split(r"(?<=[.!?。！？…])\s+", text):
        if cur and len(cur) + 1 + len(s) > limit:
            segments.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    return segments + ([cur] if cur else [])


class Breeze:
    def __init__(self, cfg: dict):
        os.environ["BREEZE_DTYPE"] = cfg["dtype"]
        if not (cfg["repo"] / "breeze_infer").is_dir():
            raise SystemExit(f"breeze-tts: Breeze source not found at {cfg['repo']} (re-run tts/setup.sh)")
        if not (cfg["weights"] / "config.json").is_file():
            raise SystemExit(f"breeze-tts: weights not found at {cfg['weights']} (re-run tts/setup.sh)")
        sys.path.insert(0, str(cfg["repo"]))
        import torch
        from breeze_infer.runtime import load_runtime, resolve_device, update_generation_config_for_breeze
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        self.torch = torch
        want = cfg["device"]                  # "cuda" means the first GPU; Breeze wants an index
        self.device = resolve_device(None if want in ("auto", "cuda") else want)
        fast = cfg["fast"] if self.device.startswith("cuda") else []   # the fast stages are CUDA/HIP graphs
        unknown = set(fast) - set(STAGES)
        if unknown:
            raise SystemExit(f"breeze-tts: unknown fast stages {sorted(unknown)}")
        t0 = time.perf_counter()
        self.tokenizer, self.model, self.audio_tokenizer = load_runtime(
            cfg["weights"], device=self.device, attn_implementation="eager")
        update_generation_config_for_breeze(self.model)
        self.runtime = FastBreezeStreamingRuntime(
            self.model, self.audio_tokenizer,
            FastStreamingConfig(max_new_tokens=1500, max_seq_len=2048, repetition_penalty=1.1,
                                **{f"fast_{s}": s in fast for s in STAGES}),
            tokenizer=self.tokenizer)
        if self.runtime.fast_enabled:
            from dataclasses import replace
            from models.warmup_profile import load_warmup_profile
            profile = load_warmup_profile(cfg["repo"] / "configs" / "fast.json")
            self.runtime.warmup_from_profile(replace(profile, codec_chunk_frames=self.runtime.codec_chunk_frames))
        self.sync()
        self.fast, self.load_s, self.sr = fast, time.perf_counter() - t0, self.runtime.sample_rate

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def report(self) -> str:
        t = self.torch
        parts = [f"torch {t.__version__}"]
        parts.append(f"ROCm/HIP {t.version.hip}" if t.version.hip else f"CUDA {t.version.cuda}" if t.version.cuda else "CPU build")
        if self.device.startswith("cuda"):
            parts.append(t.cuda.get_device_name(0))
        parts += [f"device={self.device}", f"dtype={os.environ['BREEZE_DTYPE']}", f"fast={','.join(self.fast) or 'off'}"]
        return " | ".join(parts)

    def say(self, text: str, instruction=None, ref_audio=None, ref_text=None, cfg_scale=None, seed=42):
        import numpy as np
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name
        instruction = (instruction or "").strip() or None
        cfg_scale = cfg_scale if cfg_scale is not None else (4.0 if instruction else 1.0)
        pieces = []
        for n, seg in enumerate(split_text(text)):
            req = {"id": f"s{n}", "text": seg, "speaker": "S0"}
            if instruction:
                req["instruction"] = instruction
            if ref_audio:
                req["ref_audio_path"], req["ref_text"] = str(ref_audio), (ref_text or "").strip()
            set_all_seeds(seed)
            inputs = prepare_inputs(self.tokenizer, self.audio_tokenizer, self.model, [req],
                                    get_template(select_template_name(req)), guidance_scale=cfg_scale,
                                    guidance_scale_ref=None, guidance_scale_ins=None)
            if n:
                pieces.append(np.zeros(int(GAP_S * self.sr), np.float32))
            set_all_seeds(seed)
            for chunk in self.runtime.iter_audio_chunks(inputs, request_id=req["id"], seed=seed):
                a = np.asarray(chunk.audio, dtype=np.float32)
                if not np.isfinite(a).all():
                    raise RuntimeError("Breeze produced NaN/inf audio (try BREEZE_DTYPE=fp32)")
                pieces.append(a)
        return np.concatenate(pieces) if pieces else np.zeros(1, np.float32)


def write_wav(path: Path, audio, sr: int):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def designed_reference(engine: Breeze, instruction: str, seed: int, cfg_scale) -> tuple[Path, str]:
    """Generate (once, cached) a sample of a designed voice, so every clip can clone it."""
    h = hashlib.sha1(f"{instruction}␟{seed}␟{cfg_scale}".encode()).hexdigest()[:12]
    d = DATA_DIR / "voices" / ".designed" / h
    if not (d / "reference.wav").is_file():
        log(f"  designing the voice once: {instruction!r}")
        write_wav(d / "reference.wav", engine.say(DESIGN_SAMPLE, instruction=instruction, cfg_scale=cfg_scale, seed=seed), engine.sr)
        (d / "voice.json").write_text(json.dumps({"name": f"designed-{h}", "transcript": DESIGN_SAMPLE,
                                                  "instruction": instruction, "source": "design"}, indent=1))
    return d / "reference.wav", DESIGN_SAMPLE


def resolve_voice(engine, cfg, voice=None, instruction=None, ref=None, ref_text=None, seed=42, cfg_scale=None):
    """-> kwargs for Breeze.say(): clone / direction / design-then-clone."""
    if ref:
        return {"ref_audio": Path(ref), "ref_text": ref_text, "instruction": instruction}
    if voice:
        r, t = find_voice(cfg, voice)
        return {"ref_audio": r, "ref_text": t, "instruction": instruction}
    if instruction:
        r, t = designed_reference(engine, instruction, seed, cfg_scale)
        return {"ref_audio": r, "ref_text": t}
    raise SystemExit("breeze-tts: give --voice NAME, --ref FILE --ref-text TEXT, or --instruction TEXT")


# ---------------------------------------------------------------------------------- commands

def cmd_batch(args, cfg):
    job_path = Path(args.batch).resolve()
    job, base = json.loads(job_path.read_text()), job_path.parent
    engine = Breeze(cfg)
    log(engine.report())
    log(f"loaded in {engine.load_s:.1f}s")
    seed, cfg_scale = int(job.get("seed", 42)), job.get("cfg_scale")
    ref = str(base / job["ref_audio"]) if job.get("ref_audio") else None
    voice = resolve_voice(engine, cfg, job.get("voice"), job.get("instruction"), ref, job.get("ref_text"), seed, cfg_scale)
    made, t0 = [], time.perf_counter()
    for item in job["items"]:
        v = voice if not (item.get("voice") or item.get("instruction")) else resolve_voice(
            engine, cfg, item.get("voice"), item.get("instruction"), seed=seed, cfg_scale=cfg_scale)
        audio = engine.say(item["text"], cfg_scale=cfg_scale, seed=int(item.get("seed", seed)), **v)
        out = base / item["out"]
        write_wav(out, audio, engine.sr)
        made.append((item, out, audio))
        log(f"  {item.get('id', out.name)}: {len(audio) / engine.sr:.2f}s")
    engine.sync()
    gen = time.perf_counter() - t0
    total = sum(len(a) for _, _, a in made) / engine.sr
    log(f"batch: {len(made)} clips, {total:.1f}s of audio in {gen:.1f}s (RTF {gen / max(total, 1e-6):.2f})")
    summary = timings(made, base, engine, cfg)
    print(json.dumps({"device": engine.device, "voice": job.get("voice") or job.get("instruction"), "items": summary}))


def timings(made, base, engine, cfg) -> list[dict]:
    summary = [{"id": item.get("id"), "out": str(out), "duration": round(len(a) / engine.sr, 3)} for item, out, a in made]
    todo = [k for k, (item, _, _) in enumerate(made) if item.get("timings")]
    if not todo:
        return summary
    from align import Aligner
    del engine.runtime, engine.model                    # free the GPU for Whisper
    if engine.device.startswith("cuda"):
        engine.torch.cuda.empty_cache()
    log(f"word timings: Whisper ({cfg['whisper']})")
    aligner = Aligner(cfg["whisper"], engine.device)
    for k in todo:
        item, _, audio = made[k]
        info = aligner.align(audio, engine.sr, item["text"])
        path = base / item["timings"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=1))
        summary[k]["words"] = len(info["words"])
    return summary


def cmd_say(args, cfg):
    text = Path(args.file).read_text() if args.file else " ".join(args.text)
    if not text.strip():
        raise SystemExit("breeze-tts: no text (pass it as an argument or with -f FILE)")
    job = {"voice": args.voice, "instruction": args.instruction, "ref_audio": args.ref, "ref_text": args.ref_text,
           "seed": args.seed, "cfg_scale": args.cfg,
           "items": [{"id": "say", "text": text, "out": str(Path(args.output).resolve()),
                      **({"timings": str(Path(args.timings).resolve())} if args.timings else {})}]}
    tmp = Path(args.output).resolve().with_suffix(".job.json")
    tmp.write_text(json.dumps(job))
    try:
        args.batch = str(tmp)
        cmd_batch(args, cfg)
    finally:
        tmp.unlink(missing_ok=True)


def cmd_check(args, cfg):
    engine = Breeze(cfg)
    log(engine.report())
    log(f"loaded in {engine.load_s:.1f}s")
    r, t = designed_reference(engine, "A clear, friendly narrator with a warm, even delivery.", 42, None)
    engine.say("Warming up.", ref_audio=r, ref_text=t)
    text = "Breeze is a three billion parameter speech model. If you can hear this clearly, it is working."
    t0 = time.perf_counter()
    audio = engine.say(text, ref_audio=r, ref_text=t)
    engine.sync()
    el, dur = time.perf_counter() - t0, len(audio) / engine.sr
    log(f"generated {dur:.2f}s of audio in {el:.2f}s → RTF {el / dur:.3f} (lower is faster)")
    if args.output != "out.wav":
        write_wav(Path(args.output), audio, engine.sr)
        log(f"saved {args.output}")


def cmd_add_voice(args, cfg):
    name, clip = args.add_voice
    if not args.ref_text:
        raise SystemExit("breeze-tts: --add-voice needs --ref-text with the exact words spoken in the clip")
    import soundfile as sf
    info = sf.info(clip)
    if not 3 <= info.duration <= 60:
        raise SystemExit(f"breeze-tts: the clip is {info.duration:.1f}s; use 3-60 s (10-25 s of clean speech is best)")
    d = DATA_DIR / "voices" / slug(name)
    d.mkdir(parents=True, exist_ok=True)
    import librosa
    y, _ = librosa.load(clip, sr=24000, mono=True)
    write_wav(d / "reference.wav", y, 24000)
    (d / "voice.json").write_text(json.dumps({"name": name, "transcript": args.ref_text.strip(),
                                              "duration_s": round(len(y) / 24000, 2), "source": "upload",
                                              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    log(f"saved voice {name!r} in {d}")


def main():
    p = argparse.ArgumentParser(prog="breeze-tts", description="Breeze TTS 2 with word timings (non-commercial weights).")
    p.add_argument("text", nargs="*", help="text to speak")
    p.add_argument("-f", "--file", help="read text from a file")
    p.add_argument("-o", "--output", default="out.wav", help="output WAV (default out.wav)")
    p.add_argument("--timings", help="also write word timings JSON here")
    p.add_argument("--voice", help="a saved voice name (see --voices)")
    p.add_argument("--ref", help="reference clip to clone (with --ref-text)")
    p.add_argument("--ref-text", help="exact words spoken in --ref")
    p.add_argument("--instruction", help="describe the voice (design) or, with a voice, the delivery (direction)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg", type=float, default=None, help="guidance scale (default 1, or 4 with an instruction)")
    p.add_argument("--batch", help="synthesize every item in a job JSON file")
    p.add_argument("--voices", action="store_true", help="list saved voices")
    p.add_argument("--add-voice", nargs=2, metavar=("NAME", "CLIP"), help="save a voice from a 3-60 s clip")
    p.add_argument("--check", action="store_true", help="report the device and time a test sentence")
    args = p.parse_args()
    cfg = config()
    sys.path.insert(0, str(HERE))
    if args.voices:
        for v in list_voices(cfg):
            print(f"{v['name']:24s} {v['duration_s'] or 0:5.1f}s  {v['dir']}")
    elif args.add_voice:
        cmd_add_voice(args, cfg)
    elif args.check:
        cmd_check(args, cfg)
    elif args.batch:
        cmd_batch(args, cfg)
    else:
        cmd_say(args, cfg)


if __name__ == "__main__":
    main()
