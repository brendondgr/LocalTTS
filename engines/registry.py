#!/usr/bin/env python3
"""The engine registry: ~/.config/tts-engines/engines.json. Standard library only.

Installers record what they set up here; kokoro-tts, breeze-tts, video-maker and LocalTTS read
it, so an engine installed once (for whatever GPU this machine has) is found by all of them.

  python3 registry.py show
  python3 registry.py get breeze.python
  python3 registry.py set breeze python=/x/bin/python dtype=bf16 fast=depth_decoder,backbone_decode
  python3 registry.py add-voices-dir /path/to/voices
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HOME = Path.home()
REGISTRY = Path(os.environ.get("XDG_CONFIG_HOME", HOME / ".config")) / "tts-engines" / "engines.json"
LISTS = {"fast"}                     # comma-separated on the command line, lists in the file


def load() -> dict:
    try:
        return json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        return {"version": 1, "engines": {}, "voices_dirs": []}


def save(data: dict) -> None:
    data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(REGISTRY)


def engine(name: str) -> dict:
    return load().get("engines", {}).get(name, {})


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "show":
        print(json.dumps(load(), indent=1))
        return 0
    cmd, rest = argv[0], argv[1:]
    data = load()
    if cmd == "get":
        cur = data.get("engines", {})
        for part in rest[0].split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur is None:
            return 1
        print(",".join(cur) if isinstance(cur, list) else cur)
        return 0
    if cmd == "set":
        eng = data.setdefault("engines", {}).setdefault(rest[0], {})
        for kv in rest[1:]:
            k, _, v = kv.partition("=")
            eng[k] = [x for x in v.split(",") if x] if k in LISTS else v
        eng["installed"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        save(data)
        return 0
    if cmd == "add-voices-dir":
        dirs = data.setdefault("voices_dirs", [])
        d = str(Path(rest[0]).expanduser().resolve())
        if d not in dirs:
            dirs.append(d)
            save(data)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
