#!/usr/bin/env python3
"""kokoro-tts: local text-to-speech with Kokoro-82M, with word timings.

Installed into its own environment by tts/install.sh (Linux/macOS) or tts/install.ps1 (Windows)
and run through the `kokoro-tts` launcher, so it works from any directory.

  kokoro-tts "Hello there." -o hello.wav             # one line → WAV (24 kHz mono, 16-bit)
  kokoro-tts -f script.txt -o script.wav --timings script.json
  kokoro-tts --batch job.json                        # many clips, one model load (used by video-maker)
  kokoro-tts --check                                 # device report + timed smoke test
  kokoro-tts --voices                                # list the bundled voice names

Batch job format:
  { "voice": "af_heart", "speed": 1.0, "lang": "a",
    "items": [ { "id": "hook", "text": "…", "out": "audio/hook.wav", "timings": "audio/hook.json" } ] }
Relative paths resolve against the job file's directory. Each timings file is
  { "duration": s, "sample_rate": 24000, "words": [ { "w": "Hello", "start": 0.12, "end": 0.41 } ] }
and a summary is printed to stdout as JSON.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Apple Silicon: let ops that MPS lacks fall back to the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# AMD/ROCm: reuse tuned kernels instead of re-searching for every new input length.
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")   # errors only; hides harmless "db unreadable" warnings
warnings.filterwarnings("ignore")

SAMPLE_RATE = 24_000
REPO_ID = "hexgrad/Kokoro-82M"
VOICES = {
    "a": "af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky "
         "am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa",
    "b": "bf_alice bf_emma bf_isabella bf_lily bm_daniel bm_fable bm_george bm_lewis",
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():          # NVIDIA CUDA and AMD ROCm both report as "cuda"
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_report(device: str) -> str:
    import torch
    parts = [f"torch {torch.__version__}"]
    if torch.version.hip:
        parts.append(f"ROCm/HIP {torch.version.hip}")
    elif torch.version.cuda:
        parts.append(f"CUDA {torch.version.cuda}")
    if device == "cuda":
        parts.append(torch.cuda.get_device_name(0))
    elif device == "xpu":
        parts.append(torch.xpu.get_device_name(0))
    parts.append(f"device={device}")
    return " | ".join(parts)


class Synth:
    def __init__(self, lang: str, device: str):
        from kokoro import KPipeline
        self.device = pick_device(device)
        self.pipeline = KPipeline(lang_code=lang, repo_id=REPO_ID, device=self.device)

    def say(self, text: str, voice: str, speed: float):
        """Return (float32 numpy audio, [word timings]) for the whole text."""
        import numpy as np
        chunks, words, offset = [], [], 0.0
        for r in self.pipeline(text, voice=voice, speed=speed):
            if r.audio is None:
                continue
            a = r.audio.detach().cpu().numpy().astype("float32")
            for t in r.tokens or []:
                if not any(ch.isalnum() for ch in t.text):
                    # Punctuation rides on the previous word, so captions can break on sentences.
                    if words and t.text.strip():
                        words[-1]["w"] += t.text.strip()
                    continue
                if t.start_ts is None or t.end_ts is None:
                    continue
                words.append({"w": t.text, "start": round(offset + t.start_ts, 3), "end": round(offset + t.end_ts, 3)})
            chunks.append(a)
            offset += len(a) / SAMPLE_RATE
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
        return audio, words

    def sync(self):
        import torch
        if self.device == "cuda":
            torch.cuda.synchronize()


def write_outputs(audio, words, out: Path, timings: Path | None):
    import soundfile as sf
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
    info = {"duration": round(len(audio) / SAMPLE_RATE, 3), "sample_rate": SAMPLE_RATE, "words": words}
    if timings:
        timings.parent.mkdir(parents=True, exist_ok=True)
        timings.write_text(json.dumps(info, indent=1))
    return info


def cmd_check(args):
    s = Synth(args.lang, args.device)
    log(device_report(s.device))
    s.say("Warming up.", args.voice, 1.0)          # first call compiles kernels; don't time it
    text = "Kokoro is a small text to speech model. If you can hear this clearly, the pipeline is working."
    t0 = time.perf_counter()
    audio, _ = s.say(text, args.voice, args.speed)
    s.sync()
    el = time.perf_counter() - t0
    dur = len(audio) / SAMPLE_RATE
    log(f"generated {dur:.2f}s of audio in {el:.2f}s → RTF {el / dur:.3f} (lower is faster)")
    if args.output:
        write_outputs(audio, [], Path(args.output), None)
        log(f"saved {args.output}")


def cmd_batch(args):
    job_path = Path(args.batch).resolve()
    job = json.loads(job_path.read_text())
    base = job_path.parent
    voice, speed, lang = job.get("voice", args.voice), float(job.get("speed", args.speed)), job.get("lang", args.lang)
    s = Synth(lang, job.get("device", args.device))
    log(device_report(s.device))
    s.say("Warming up.", voice, 1.0)
    summary, t0 = [], time.perf_counter()
    for item in job["items"]:
        out = base / item["out"]
        tim = base / item["timings"] if item.get("timings") else None
        audio, words = s.say(item["text"], item.get("voice", voice), float(item.get("speed", speed)))
        info = write_outputs(audio, words, out, tim)
        summary.append({"id": item.get("id"), "out": str(out), "duration": info["duration"], "words": len(words)})
        log(f"  {item.get('id', out.name)}: {info['duration']:.2f}s, {len(words)} words")
    s.sync()
    total = sum(x["duration"] for x in summary)
    log(f"batch: {len(summary)} clips, {total:.1f}s of audio in {time.perf_counter() - t0:.1f}s")
    print(json.dumps({"device": s.device, "voice": voice, "speed": speed, "items": summary}))


def cmd_say(args):
    text = Path(args.file).read_text() if args.file else " ".join(args.text)
    if not text.strip():
        sys.exit("kokoro-tts: no text (pass it as an argument or with -f FILE)")
    s = Synth(args.lang, args.device)
    log(device_report(s.device))
    audio, words = s.say(text, args.voice, args.speed)
    info = write_outputs(audio, words, Path(args.output), Path(args.timings) if args.timings else None)
    log(f"saved {args.output} ({info['duration']:.2f}s)")


def main():
    p = argparse.ArgumentParser(prog="kokoro-tts", description="Local Kokoro-82M text-to-speech with word timings.")
    p.add_argument("text", nargs="*", help="text to speak")
    p.add_argument("-f", "--file", help="read text from a file")
    p.add_argument("-o", "--output", default="out.wav", help="output WAV (default out.wav)")
    p.add_argument("--timings", help="also write word timings JSON here")
    p.add_argument("--voice", default="af_heart", help="voice name (default af_heart); see --voices")
    p.add_argument("--speed", type=float, default=1.0, help="speaking rate multiplier (default 1.0)")
    p.add_argument("--lang", default="a", help="a = American English, b = British English (default a)")
    p.add_argument("--device", default="auto", help="auto | cuda | xpu | mps | cpu (ROCm counts as cuda)")
    p.add_argument("--batch", help="synthesize every item in a job JSON file")
    p.add_argument("--check", action="store_true", help="report the device and time a test sentence")
    p.add_argument("--voices", action="store_true", help="list English voice names")
    args = p.parse_args()
    if args.voices:
        for k, v in VOICES.items():
            print(f"lang {k}: {v}")
        print("More voices/languages: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md")
    elif args.check:
        args.output = args.output if args.output != "out.wav" else None
        cmd_check(args)
    elif args.batch:
        cmd_batch(args)
    else:
        cmd_say(args)


if __name__ == "__main__":
    main()
