"""Settings, all overridable through the environment (see localtts.env.example)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Engines installed by engines/setup.sh (the same installer video-maker uses) are recorded here,
# with the backend, dtype and fast stages chosen for this machine's GPU. Environment variables
# still override every value.
REGISTRY = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tts-engines" / "engines.json"


def _registry() -> dict:
    try:
        return json.loads(REGISTRY.read_text())
    except (OSError, ValueError):
        return {}


_REG = _registry()
_ENG = _REG.get("engines", {})


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _engine_python(env: str, engine: str, *fallbacks: str) -> Path:
    """$env, else the registry, else the first fallback that exists (else the first one)."""
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    if _ENG.get(engine, {}).get("python"):
        return Path(_ENG[engine]["python"])
    paths = [Path(f).expanduser() for f in fallbacks]
    return next((p for p in paths if p.exists()), paths[0])


def breeze_env() -> dict[str, str]:
    """Settings for the Breeze worker: registry values unless the environment sets them."""
    b = _ENG.get("breeze", {})
    defaults = {
        "BREEZE_REPO": b.get("repo", str(Path("~/Projects/breeze-tts").expanduser())),
        "BREEZE_MODEL": b.get("weights", str(Path("~/models/breeze-tts-2").expanduser())),
        "BREEZE_DTYPE": b.get("dtype", "bf16"),
        "BREEZE_DEVICE": b.get("device", "auto"),
        "LOCALTTS_BREEZE_FAST": ",".join(b["fast"]) if "fast" in b else "depth_decoder,backbone_decode",
        "LOCALTTS_WHISPER_MODEL": b.get("whisper", "openai/whisper-large-v3-turbo"),
    }
    return {k: os.environ.get(k, v) for k, v in defaults.items()}


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("LOCALTTS_HOST", "127.0.0.1")
    port: int = int(os.environ.get("LOCALTTS_PORT", "5040"))
    data: Path = _path("LOCALTTS_DATA", str(ROOT / "data"))
    outputs: Path = _path("LOCALTTS_OUTPUTS_DIR", "~/Music/TTS")
    keep_alive_s: int = int(os.environ.get("LOCALTTS_KEEP_ALIVE_S", "330"))  # > any browser's idle pool
    idle_unload_s: float = float(os.environ.get("LOCALTTS_IDLE_UNLOAD_S", "600"))
    load_timeout_s: float = float(os.environ.get("LOCALTTS_LOAD_TIMEOUT_S", "900"))
    breeze_python: Path = _engine_python("LOCALTTS_BREEZE_PYTHON", "breeze",
                                         "~/venvs/breeze-tts/bin/python", "~/venvs/breeze-next/bin/python")
    kokoro_python: Path = _engine_python("LOCALTTS_KOKORO_PYTHON", "kokoro", "~/venvs/kokoro/bin/python")
    default_engine: str = os.environ.get("LOCALTTS_DEFAULT_ENGINE", "kokoro")
    default_kokoro_voice: str = os.environ.get("LOCALTTS_DEFAULT_KOKORO_VOICE", "af_heart")
    # AI-Enhance: any OpenAI-compatible chat endpoint (default: the DashLLM relay on this machine).
    llm_url: str = os.environ.get("LOCALTTS_LLM_URL", "http://127.0.0.1:4000/v1").rstrip("/")
    llm_model: str = os.environ.get("LOCALTTS_LLM_MODEL", "auto")
    llm_api_key: str = os.environ.get("LOCALTTS_LLM_API_KEY", "")
    llm_timeout_s: float = float(os.environ.get("LOCALTTS_LLM_TIMEOUT_S", "120"))
    preload: tuple[str, ...] = field(default_factory=lambda: tuple(
        e for e in os.environ.get("LOCALTTS_PRELOAD", "").split(",") if e))

    @property
    def voices_dir(self) -> Path:
        return self.data / "voices"

    @property
    def outputs_dir(self) -> Path:
        return self.outputs


settings = Settings()
