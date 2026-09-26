#!/usr/bin/env python3
"""Look at this machine and decide how each TTS engine should be installed. Read-only.

  python3 tts/probe.py            # human-readable report + plan
  python3 tts/probe.py --json     # the same as JSON (what setup.py and Claude read)

Nothing here installs or changes anything; setup.py acts on the plan. Standard library only,
so it runs before any environment exists (Python 3.8+).
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HOME = Path.home()
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")) / "tts-engines"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", HOME / ".local" / "share")) / "tts-engines"
REGISTRY = CONFIG_DIR / "engines.json"
BIN_DIR = HOME / ".local" / "bin"

KOKORO_HOME = Path(os.environ.get("KOKORO_HOME", HOME / "venvs" / "kokoro"))
BREEZE_HOME = Path(os.environ.get("BREEZE_HOME", HOME / "venvs" / "breeze-tts"))
BREEZE_UPSTREAM = "https://github.com/breezeblue-ai/breeze-tts.git"
BREEZE_WEIGHTS_REPO = "BreezeBlue/Breeze-TTS-2"
BREEZE_TORCH = "torch==2.9.1 torchaudio==2.9.1"          # what upstream pins and tests
LOCALTTS_REPO = "https://github.com/brendondgr/LocalTTS.git"
LOCALTTS_DIR = Path(os.environ.get("LOCALTTS_DIR", DATA_DIR / "LocalTTS"))
GIB = 1024 ** 3

# Sizes for the plan (GB). Measured on the reference install.
SIZE = {"kokoro_env": 6.5, "kokoro_model": 0.35, "breeze_env": 7.0, "breeze_weights": 7.2, "whisper": 1.6}
BREEZE_MIN_GPU_GIB = 10      # peak 8.5 GiB at bf16 during generation, plus headroom


def run(*cmd: str, timeout: float = 15) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def which(name: str) -> str | None:
    return shutil.which(name)


def free_gb(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1e9, 1)
    except OSError:
        return -1.0


# ---------------------------------------------------------------------------------- hardware

def nvidia_gpus() -> list[dict]:
    if not which("nvidia-smi"):
        return []
    out = run("nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
              "--format=csv,noheader,nounits")
    head = run("nvidia-smi")
    m = re.search(r"CUDA Version:\s*([\d.]+)", head)
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, mem, cc, drv = parts[:4]
        try:
            cc_f = float(cc)
        except ValueError:
            cc_f = 0.0
        gpus.append({"vendor": "nvidia", "name": name, "memory_gib": round(float(mem) / 1024, 1),
                     "compute_capability": cc_f, "driver": drv, "cuda_driver": m.group(1) if m else None,
                     "unified": False})
    return gpus


def amd_gfx_targets() -> list[str]:
    targets = []
    for props in sorted(Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties")):
        try:
            text = props.read_text()
        except OSError:
            continue
        m = re.search(r"gfx_target_version\s+(\d+)", text)
        if m and int(m.group(1)):
            v = int(m.group(1))
            targets.append(f"gfx{v // 10000}{(v // 100) % 100}{v % 100:x}")
    if not targets and which("rocminfo"):
        targets = sorted(set(t for t in re.findall(r"gfx[0-9a-f]+", run("rocminfo")) if t != "gfx000"))
    return targets


def amd_gpus() -> list[dict]:
    targets = amd_gfx_targets()
    if not targets:
        return []
    names = re.findall(r"Marketing Name:\s*(.+)", run("rocminfo")) if which("rocminfo") else []
    names = [n.strip() for n in names if "CPU" not in n and "Processor" not in n]
    mem = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
        try:
            if card.joinpath("vendor").read_text().strip() != "0x1002":
                continue
            vram = int(card.joinpath("mem_info_vram_total").read_text()) / GIB
            gtt = int(card.joinpath("mem_info_gtt_total").read_text()) / GIB
        except (OSError, ValueError):
            continue
        mem.append((vram, gtt))
    gpus = []
    for i, gfx in enumerate(targets):
        vram, gtt = mem[i] if i < len(mem) else (0.0, 0.0)
        # APUs (e.g. Strix Halo) have a small or BIOS-set VRAM carve-out and use system RAM via GTT.
        unified = gfx.startswith("gfx115") or gfx in ("gfx1103", "gfx1036", "gfx1035") or vram < 4 <= gtt
        usable = max(vram, gtt) if unified else vram
        gpus.append({"vendor": "amd", "name": names[i] if i < len(names) else f"AMD {gfx}", "gfx": gfx,
                     "memory_gib": round(usable, 1), "vram_gib": round(vram, 1), "gtt_gib": round(gtt, 1),
                     "unified": unified})
    return gpus


def apple_gpu() -> list[dict]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    mem = int(run("sysctl", "-n", "hw.memsize") or 0) / GIB
    return [{"vendor": "apple", "name": run("sysctl", "-n", "machdep.cpu.brand_string") or "Apple Silicon",
             "memory_gib": round(mem, 1), "unified": True}]


def ram_gib() -> float:
    try:
        m = re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())
        return round(int(m.group(1)) / 1024 / 1024, 1)
    except (OSError, AttributeError):
        v = run("sysctl", "-n", "hw.memsize")
        return round(int(v) / GIB, 1) if v.isdigit() else 0.0


# ---------------------------------------------------------------------------------- existing installs

def read_registry() -> dict:
    try:
        return json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        return {}


def torch_info(python: Path) -> dict | None:
    if not python.exists():
        return None
    code = ("import json,torch;print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,"
            "'cuda':torch.version.cuda,'gpu':torch.cuda.is_available()}))")
    out = run(str(python), "-c", code, timeout=60)
    try:
        return json.loads(out.splitlines()[-1])
    except (ValueError, IndexError):
        return {"broken": True}


def find_breeze_weights(reg: dict) -> str | None:
    candidates = [reg.get("engines", {}).get("breeze", {}).get("weights"),
                  str(DATA_DIR / "breeze-tts-2"), str(HOME / "models" / "breeze-tts-2")]
    hf = HOME / ".cache" / "huggingface" / "hub" / "models--BreezeBlue--Breeze-TTS-2" / "snapshots"
    if hf.is_dir():
        candidates += [str(p) for p in sorted(hf.iterdir())]
    for c in candidates:
        if c and Path(c, "config.json").is_file() and list(Path(c).glob("*.safetensors")):
            return c
    return None


def find_localtts() -> str | None:
    """An installed LocalTTS checkout: via the `localtts` command, else the usual places."""
    exe = which("localtts")
    cands = [Path(exe).resolve().parent.parent] if exe else []
    cands += [LOCALTTS_DIR, HOME / "Projects" / "LocalTTS", HOME / "LocalTTS"]
    for c in cands:
        if (c / "localtts" / "server.py").is_file():
            return str(c)
    return None


def localtts_health() -> dict | None:
    url = os.environ.get("LOCALTTS_URL", "http://127.0.0.1:5040").rstrip("/")
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            return {"url": url, **json.load(r)}
    except Exception:
        return None


# ---------------------------------------------------------------------------------- the plan

ROCM_FAMILY = [  # AMD TheRock nightly indexes are per GPU family (same table as install.sh)
    (r"gfx115[0-3]|gfx90[068]|gfx90a", None), (r"gfx110.", "gfx110X-all"), (r"gfx120.", "gfx120X-all"),
    (r"gfx103.", "gfx103X-all"), (r"gfx101.", "gfx101X-dgpu"), (r"gfx94.", "gfx94X-dcgpu"), (r"gfx950", "gfx950-dcgpu"),
]


def rocm_family(gfx: str) -> str | None:
    for pat, fam in ROCM_FAMILY:
        if re.fullmatch(pat, gfx):
            return fam or gfx
    return None


def cuda_flavour(cuda_driver: str | None) -> str | None:
    """Newest PyTorch 2.9 CUDA wheel the installed driver can run."""
    try:
        v = tuple(int(x) for x in (cuda_driver or "").split(".")[:2])
    except ValueError:
        return None
    for need, flav in (((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126")):
        if v >= need:
            return flav
    return None


def plan_kokoro(sysinfo: dict, reg: dict) -> dict:
    gpu = sysinfo["gpu"]
    p = {"engine": "kokoro", "notes": [], "warnings": [], "home": str(KOKORO_HOME)}
    backend, args = "cpu", ["--gpu", "cpu"]
    if gpu and gpu["vendor"] == "nvidia":
        backend, args = "cuda", ["--gpu", "cuda", "--cuda", cuda_flavour(gpu.get("cuda_driver")) or "cu126"]
    elif gpu and gpu["vendor"] == "amd":
        backend, args = "rocm", ["--gpu", "rocm", "--rocm-arch", gpu["gfx"]]
        if not rocm_family(gpu["gfx"]):
            p["warnings"].append(f"no ROCm nightly index is known for {gpu['gfx']}; falling back to CPU")
            backend, args = "cpu", ["--gpu", "cpu"]
    elif gpu and gpu["vendor"] == "apple":
        backend, args = "mps", ["--gpu", "mps"]
    p.update(backend=backend, install_args=args, download_gb=SIZE["kokoro_env"] + SIZE["kokoro_model"])
    if backend == "cpu":
        p["notes"].append("CPU is fine for Kokoro (82M parameters): roughly real time or faster")
    existing = sysinfo["existing"]["kokoro"]
    if existing.get("torch") and not existing["torch"].get("broken"):
        p["status"] = "installed"
        p["notes"].append(f"already installed at {existing['home']} (torch {existing['torch']['torch']})")
    else:
        p["status"] = "install"
    if not sysinfo["tools"]["espeak-ng"]:
        p["warnings"].append("espeak-ng is missing: Kokoro bundles a copy, but the system package handles unknown words "
                             "better (dnf/apt/pacman install espeak-ng, brew install espeak-ng)")
    return p


def plan_breeze(sysinfo: dict, reg: dict) -> dict:
    gpu = sysinfo["gpu"]
    p = {"engine": "breeze", "notes": [], "warnings": [], "home": str(BREEZE_HOME),
         "src": str(DATA_DIR / "breeze-tts"), "weights": sysinfo["existing"]["breeze"].get("weights")
         or str(DATA_DIR / "breeze-tts-2"), "license": "BreezeBlue research & non-commercial licence (weights and outputs)"}
    fast = ["depth_decoder", "backbone_decode"]
    if gpu and gpu["vendor"] == "nvidia":
        cc = gpu.get("compute_capability") or 0
        flav = cuda_flavour(gpu.get("cuda_driver"))
        if cc < 7.0:
            return {**p, "status": "unsupported", "reason": f"{gpu['name']} (sm_{int(cc * 10)}) is too old for Breeze"}
        if not flav:
            return {**p, "status": "unsupported", "reason": f"NVIDIA driver supports CUDA {gpu.get('cuda_driver')}; "
                                                          "update the driver to one supporting CUDA 12.6 or newer"}
        dtype = "bf16" if cc >= 8.0 else "fp16"
        p.update(backend="cuda", device="cuda", dtype=dtype, fast=fast, tested=False,
                 torch_index=f"https://download.pytorch.org/whl/{flav}", torch_spec=BREEZE_TORCH)
        p["notes"].append(f"PyTorch 2.9.1 {flav} (the version upstream tests), {dtype}, CUDA-graph fast stages")
        if dtype == "fp16":
            p["warnings"].append(f"{gpu['name']} has no native bf16 (sm_{int(cc * 10)}), so Breeze runs in fp16. "
                                 "This is untested: check the smoke test sounds right; BREEZE_DTYPE=fp32 is the fallback")
        else:
            p["warnings"].append("the CUDA path follows upstream but has not been run through this installer yet")
    elif gpu and gpu["vendor"] == "amd":
        gfx = gpu["gfx"]
        if gfx == "gfx1151":
            # Tested: AMD's whl-next build defaults to hipBLASLt and supports HIP-graph capture.
            p.update(backend="rocm", device="cuda", dtype="bf16", fast=fast, tested=True,
                     torch_index="https://nightly.repo.amd.com/rocm/whl-next/", torch_spec="torch[device-gfx1151] torchaudio")
            p["notes"].append("AMD whl-next nightly PyTorch (tested on this chip: RTF about 1.5 with fast stages)")
        elif rocm_family(gfx):
            p.update(backend="rocm", device="cuda", dtype="bf16", fast=[], tested=False,
                     torch_index=f"https://rocm.nightlies.amd.com/v2/{rocm_family(gfx)}/", torch_spec="torch torchaudio")
            p["warnings"].append(f"{gfx} is untested. Fast stages are off: on the TheRock builds tried so far, "
                                 "hipBLASLt could not be captured in a HIP graph and the fast path hung. Eager mode works "
                                 "but is slower (RTF about 2.7 on gfx1151). Try --breeze-fast later if you want.")
        else:
            return {**p, "status": "unsupported", "reason": f"no PyTorch ROCm build is known for {gfx}"}
        if not sysinfo["kfd_access"]:
            p["warnings"].append("/dev/kfd is not readable/writable: sudo usermod -aG render,video $USER, then log in again")
    elif gpu and gpu["vendor"] == "apple":
        p.update(backend="cpu", device="cpu", dtype="bf16", fast=[], tested=False, torch_index=None, torch_spec=BREEZE_TORCH)
        p["warnings"].append("Breeze does not support Apple's GPU (MPS); it would run on the CPU, very slowly "
                             "(expect many times slower than real time). Use Kokoro, or LocalTTS on a GPU machine.")
        p["needs_flag"] = "--allow-cpu-breeze"
    else:
        p.update(backend="cpu", device="cpu", dtype="bf16", fast=[], tested=False,
                 torch_index="https://download.pytorch.org/whl/cpu", torch_spec=BREEZE_TORCH)
        p["warnings"].append("no supported GPU found: Breeze (3B parameters) would run on the CPU, very slowly. "
                             "Kokoro is the better choice here, or LocalTTS on a GPU machine.")
        p["needs_flag"] = "--allow-cpu-breeze"
    if gpu and gpu.get("memory_gib", 0) and gpu["memory_gib"] < BREEZE_MIN_GPU_GIB and p.get("device") != "cpu":
        return {**p, "status": "unsupported",
                "reason": f"{gpu['name']} has {gpu['memory_gib']} GiB; Breeze needs about {BREEZE_MIN_GPU_GIB} GiB"}
    p["whisper"] = "openai/whisper-large-v3-turbo" if p.get("device") != "cpu" else "openai/whisper-base.en"
    existing = sysinfo["existing"]["breeze"]
    need = SIZE["breeze_env"] + (0 if existing.get("weights") else SIZE["breeze_weights"]) + SIZE["whisper"]
    p["download_gb"] = round(need, 1)
    if existing.get("weights"):
        p["notes"].append(f"reusing the weights already at {existing['weights']}")
    if existing.get("registered") and existing.get("torch") and not existing["torch"].get("broken"):
        p["status"] = "installed"
        p["notes"].append(f"already installed ({existing['python']}, torch {existing['torch']['torch']})")
    else:
        p["status"] = "install"
    if not sysinfo["tools"]["git"]:
        p["status"], p["reason"] = "blocked", "git is required to fetch the Breeze source"
    return p


def plan_localtts(sysinfo: dict) -> dict:
    ex = sysinfo["existing"]["localtts"]
    p = {"engine": "localtts", "notes": [], "warnings": [], "dir": ex.get("dir") or str(LOCALTTS_DIR),
         "repo": LOCALTTS_REPO, "download_gb": 0.3}
    service = {"Linux": "starts at boot (systemd user service)", "Darwin": "starts at login (LaunchAgent; untested)"}
    p["notes"].append("optional app: an always-on API + web UI for these engines, freeing the GPU after 10 idle "
                      "minutes; video-maker's default provider uses it whenever it is running")
    if ex.get("health"):
        p["status"] = "installed"
        p["notes"].append(f"running at {ex['health']['url']} from {p['dir']}")
    elif ex.get("dir"):
        p["status"] = "installed"
        p["warnings"].append(f"installed at {p['dir']} but not running: localtts start")
    else:
        p["status"] = "install"
        p["notes"].append(f"{service.get(sysinfo['os'], 'runs in the background with `localtts start`')}")
        if not sysinfo["tools"]["git"]:
            p["status"], p["reason"] = "blocked", "git is required to fetch LocalTTS"
    return p


def probe() -> dict:
    reg = read_registry()
    gpus = nvidia_gpus() or amd_gpus() or apple_gpu()
    gpu = max(gpus, key=lambda g: g.get("memory_gib", 0)) if gpus else None
    in_container = Path("/run/.toolboxenv").exists() or Path("/run/.containerenv").exists()
    kreg = reg.get("engines", {}).get("kokoro", {})
    k_home = Path(kreg.get("home", KOKORO_HOME)).expanduser()
    breg = reg.get("engines", {}).get("breeze", {})
    b_python = Path(breg.get("python", BREEZE_HOME / "bin" / "python")).expanduser()
    sysinfo = {
        "os": platform.system(), "arch": platform.machine(), "release": platform.release(),
        "container": in_container, "ram_gib": ram_gib(), "gpus": gpus, "gpu": gpu,
        "kfd_access": os.access("/dev/kfd", os.R_OK | os.W_OK),
        "tools": {t: bool(which(t)) for t in ("uv", "git", "ffmpeg", "espeak-ng", "curl")},
        "disk_free_gb": {"venvs": free_gb(BREEZE_HOME), "data": free_gb(DATA_DIR)},
        "registry": str(REGISTRY),
        "existing": {
            "kokoro": {"home": str(k_home), "launcher": which("kokoro-tts"), "torch": torch_info(k_home / "bin" / "python")},
            "breeze": {"python": str(b_python), "registered": bool(breg), "launcher": which("breeze-tts"),
                       "torch": torch_info(b_python) if breg else None, "weights": find_breeze_weights(reg)},
            "localtts": {"dir": find_localtts(), "health": localtts_health()},
        },
    }
    plans = {"kokoro": plan_kokoro(sysinfo, reg), "breeze": plan_breeze(sysinfo, reg), "localtts": plan_localtts(sysinfo)}
    blockers = []
    if not sysinfo["tools"]["uv"]:
        blockers.append("uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh")
    if sysinfo["os"] == "Windows":
        blockers.append("Windows: use tts/install.ps1 (Kokoro only). Breeze and this planner are Linux/macOS")
    return {"system": sysinfo, "plans": plans, "blockers": blockers}


def show(report: dict) -> None:
    s, out = report["system"], []
    g = s["gpu"]
    out.append("System")
    out.append(f"  {s['os']} {s['arch']} ({s['release']}){'  · inside a container' if s['container'] else ''}, "
               f"{s['ram_gib']} GiB RAM")
    if g:
        extra = f", {g['gfx']}" if g.get("gfx") else f", sm_{int(g['compute_capability'] * 10)}" if g.get("compute_capability") else ""
        drv = f", driver {g['driver']} (CUDA {g['cuda_driver']})" if g.get("driver") else ""
        out.append(f"  GPU: {g['name']}{extra}, {g['memory_gib']} GiB{' unified' if g.get('unified') else ''}{drv}")
    else:
        out.append("  GPU: none usable found (CPU only)")
    out.append("  tools: " + ", ".join(f"{t} {'✔' if ok else '✖'}" for t, ok in s["tools"].items()))
    out.append(f"  free disk: {s['disk_free_gb']['venvs']} GB (venvs), {s['disk_free_gb']['data']} GB (models)")

    for name, p in report["plans"].items():
        out.append("")
        status = {"installed": "installed ✔", "install": "will install", "unsupported": "not supported here",
                  "blocked": "blocked"}[p["status"]]
        out.append(f"{ {'localtts': 'LocalTTS'}.get(name, name.capitalize()) }: {status}")
        if p.get("reason"):
            out.append(f"  reason: {p['reason']}")
        if p["status"] in ("install", "installed"):
            if name == "localtts":
                out.append(f"  {p['dir']}" + (f"  (from {p['repo']})" if p["status"] == "install" else ""))
            elif name == "kokoro":
                out.append(f"  backend {p['backend']}  →  {p['home']}")
            else:
                out.append(f"  backend {p['backend']}, {p['dtype']}, fast stages: {', '.join(p['fast']) or 'off (eager)'}"
                           f"{'' if p.get('tested') else '  [untested combination]'}")
                out.append(f"  torch: {p['torch_spec']}" + (f" from {p['torch_index']}" if p.get("torch_index") else ""))
                out.append(f"  env {p['home']}, source {p['src']}, weights {p['weights']}")
                out.append(f"  licence: {p['license']}")
            if p["status"] == "install":
                out.append(f"  download ≈ {p['download_gb']} GB")
        for n in p["notes"]:
            out.append(f"  · {n}")
        for w in p["warnings"]:
            out.append(f"  ! {w}")
        if p.get("needs_flag") and p["status"] == "install":
            out.append(f"  (installs only with {p['needs_flag']})")
    if report["blockers"]:
        out.append("")
        out += [f"✖ {b}" for b in report["blockers"]]
    print("\n".join(out))


if __name__ == "__main__":
    r = probe()
    if "--json" in sys.argv:
        print(json.dumps(r, indent=1))
    else:
        show(r)
