"""Settings, all overridable through the environment (see localtts.env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("LOCALTTS_HOST", "127.0.0.1")
    port: int = int(os.environ.get("LOCALTTS_PORT", "5040"))
    data: Path = _path("LOCALTTS_DATA", str(ROOT / "data"))
    outputs: Path = _path("LOCALTTS_OUTPUTS_DIR", "~/Music/TTS")
    keep_alive_s: int = int(os.environ.get("LOCALTTS_KEEP_ALIVE_S", "330"))  # > any browser's idle pool
    idle_unload_s: float = float(os.environ.get("LOCALTTS_IDLE_UNLOAD_S", "600"))
    load_timeout_s: float = float(os.environ.get("LOCALTTS_LOAD_TIMEOUT_S", "900"))
    breeze_python: Path = _path("LOCALTTS_BREEZE_PYTHON", "~/venvs/breeze-next/bin/python")
    kokoro_python: Path = _path("LOCALTTS_KOKORO_PYTHON", "~/venvs/kokoro/bin/python")
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
