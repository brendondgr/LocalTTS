#!/usr/bin/env python3
"""Set up the TTS engines for this machine: look first, explain the plan, then install.

  bash tts/setup.sh --plan                     # only look and explain (changes nothing)
  bash tts/setup.sh                            # look, explain, ask, install
  bash tts/setup.sh --engines kokoro           # just Kokoro
  bash tts/setup.sh --yes --accept-breeze-license

Options:
  --plan / --json            show the plan (human / JSON) and stop
  --engines kokoro,breeze,localtts   what to consider (default: all three; LocalTTS is the optional server app)
  --yes                      don't ask before installing
  --accept-breeze-license    accept the Breeze TTS 2 research / non-commercial licence
  --allow-cpu-breeze         install Breeze even without a supported GPU (very slow)
  --breeze-fast a,b          override the fast stages ("" = eager)
  --breeze-dtype bf16|fp16|fp32
  --reinstall                re-run an engine's installer even if it is already installed
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe      # noqa: E402
import registry   # noqa: E402


def ask(question: str, default: bool) -> bool:
    if not sys.stdin.isatty():
        return default
    hint = "Y/n" if default else "y/N"
    ans = input(f"{question} [{hint}] ").strip().lower()
    return default if not ans else ans.startswith("y")


def adopt_existing(report: dict) -> None:
    """Record an engine that is installed but not in the registry yet (e.g. an older Kokoro)."""
    k = report["system"]["existing"]["kokoro"]
    if k.get("torch") and not k["torch"].get("broken") and not registry.engine("kokoro"):
        registry.main(["set", "kokoro", f"python={k['home']}/bin/python", f"home={k['home']}",
                       f"backend={report['plans']['kokoro']['backend']}", f"torch={k['torch']['torch']}",
                       f"launcher={k.get('launcher') or ''}"])
        print(f"» recorded the existing Kokoro install ({k['home']}) in {registry.REGISTRY}")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--engines", default="kokoro,breeze,localtts")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--accept-breeze-license", action="store_true")
    ap.add_argument("--allow-cpu-breeze", action="store_true")
    ap.add_argument("--breeze-fast")
    ap.add_argument("--breeze-dtype")
    ap.add_argument("--reinstall", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    a = ap.parse_args()
    if a.help:
        print(__doc__)
        return 0

    report = probe.probe()
    if a.json:
        print(json.dumps(report, indent=1))
        return 0
    probe.show(report)
    if a.plan:
        return 0
    if report["blockers"]:
        print("\nFix the ✖ items above, then run this again.")
        return 1

    wanted = [e.strip() for e in a.engines.split(",") if e.strip()]
    todo = []
    for name in wanted:
        p = report["plans"].get(name)
        if not p:
            print(f"! unknown engine {name!r}")
            continue
        if p["status"] == "installed" and not a.reinstall:
            continue
        if p["status"] in ("unsupported", "blocked"):
            print(f"\n» skipping {name}: {p.get('reason')}")
            continue
        if name == "breeze":
            if p.get("needs_flag") and not a.allow_cpu_breeze:
                print(f"\n» skipping Breeze: no supported GPU (pass {p['needs_flag']} to install it on the CPU anyway)")
                continue
            if not a.accept_breeze_license:
                print("\nBreeze TTS 2's weights, and all audio they generate, are licensed for research and\n"
                      "non-commercial use only (https://huggingface.co/BreezeBlue/Breeze-TTS-2).")
                if not ask("Accept that licence and install Breeze?", False):
                    print("» skipping Breeze")
                    continue
        if name == "localtts" and not a.yes and not ask(
                "Install LocalTTS too (an always-on TTS server + web UI that video-maker prefers when running)?", True):
            print("» skipping LocalTTS")
            continue
        todo.append((name, p))

    adopt_existing(report)
    if not todo:
        print("\nNothing to install.")
        return 0
    print("\nWill install: " + ", ".join(f"{n} (≈{p['download_gb']} GB download)" for n, p in todo))
    if not a.yes and not ask("Go ahead?", True):
        return 1

    for name, p in todo:
        if name == "localtts":
            d = Path(p["dir"])
            if not (d / "install.sh").is_file():
                print(f"\n━━ fetching LocalTTS ━━\n$ git clone {p['repo']} {d}")
                d.parent.mkdir(parents=True, exist_ok=True)
                if subprocess.call(["git", "clone", "--quiet", p["repo"], str(d)]) != 0:
                    print("✖ could not clone LocalTTS")
                    return 1
            cmd = ["bash", str(d / "install.sh"), "--no-engines"]   # engines are handled above
        elif name == "kokoro":
            cmd = ["bash", str(HERE / "install.sh"), *p["install_args"]] + (["--force-torch"] if a.reinstall else [])
        else:
            fast = a.breeze_fast if a.breeze_fast is not None else ",".join(p["fast"])
            cmd = ["bash", str(HERE / "breeze" / "install.sh"), "--backend", p["backend"], "--device", p["device"],
                   "--dtype", a.breeze_dtype or p["dtype"], "--fast", fast, "--whisper", p["whisper"],
                   "--torch-spec", p["torch_spec"], "--prefix", p["home"], "--src", p["src"],
                   "--weights", p["weights"], "--accept-license"]
            if p.get("torch_index"):
                cmd += ["--torch-index", p["torch_index"]]
        print(f"\n━━ installing {name} ━━\n$ {' '.join(cmd)}")
        if subprocess.call(cmd) != 0:
            print(f"✖ {name} failed; the output above says why. Re-running continues where it stopped.")
            return 1

    print("\nInstalled. Engines recorded in", registry.REGISTRY)
    for n, e in registry.load().get("engines", {}).items():
        print(f"  {n:7s} {e.get('backend', '?'):5s} torch {e.get('torch', '?')}  → {e.get('launcher') or e.get('python')}")
    if not any(n == "localtts" for n, _ in todo) and report["plans"]["localtts"]["status"] == "install":
        print("\nOptional: LocalTTS runs these engines as an always-on API + web UI (bash tts/setup.sh --engines localtts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
