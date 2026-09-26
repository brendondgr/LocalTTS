"""Engine workers: one subprocess per engine, started on demand, stopped when idle.

Stopping the process is the unload. It hands back every byte of GPU memory (weights,
the caching allocator, captured graphs), which an in-process `del model` does not
reliably do.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path

from .config import ROOT, breeze_env, settings

log = logging.getLogger("localtts.engines")
WORKERS = ROOT / "workers"
DRAIN_TIMEOUT_S = 30.0  # longest we wait for an abandoned request before restarting the worker


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, name: str, python: Path, script: str, env: dict[str, str] | None = None):
        self.name = name
        self.python = python
        self.script = WORKERS / script
        self.env = env or {}
        self.proc: asyncio.subprocess.Process | None = None
        self.info: dict = {}
        self.state = "unloaded"  # unloaded | loading | ready | busy
        self.last_used = time.monotonic()
        self.loaded_at: float | None = None
        self._lock = asyncio.Lock()        # one request at a time per engine
        self._load_lock = asyncio.Lock()
        self._stale = False  # an abandoned request is still emitting; read it off before the next

    # ---- lifecycle -------------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def ensure_loaded(self) -> None:
        async with self._load_lock:
            if self.alive and self.state != "loading":
                return
            if not self.python.exists():
                raise EngineError(f"{self.name} is not installed (no Python at {self.python}); set it up with: bash engines/setup.sh --plan")
            self.state = "loading"
            log.info("%s: starting worker", self.name)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", **self.env}
            self.proc = await asyncio.create_subprocess_exec(
                str(self.python), str(self.script),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=None, cwd=str(WORKERS), env=env,
                limit=1 << 20, start_new_session=True,
            )
            try:
                msg = await asyncio.wait_for(self._read_event(), settings.load_timeout_s)
            except Exception:
                await self.unload(reason="failed to load")
                raise
            if msg.get("event") != "ready":
                await self.unload(reason="failed to load")
                raise EngineError(f"{self.name}: failed to load: {msg.get('message', msg)}")
            self.info = {k: v for k, v in msg.items() if k != "event"}
            self.state = "ready"
            self.loaded_at = time.monotonic()
            self.last_used = time.monotonic()
            log.info("%s: ready %s", self.name, self.info)

    async def unload(self, reason: str = "requested") -> bool:
        proc, self.proc, self._stale = self.proc, None, False
        self.state, self.info, self.loaded_at = "unloaded", {}, None
        if proc is None or proc.returncode is not None:
            return False
        log.info("%s: unloading (%s), pid %s", self.name, reason, proc.pid)
        try:
            proc.stdin.close()
            await asyncio.wait_for(proc.wait(), 10)
        except (asyncio.TimeoutError, ProcessLookupError, BrokenPipeError, ConnectionResetError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
        return True

    async def maybe_idle_unload(self) -> None:
        if (self.alive and self.state == "ready" and not self._lock.locked()
                and time.monotonic() - self.last_used > settings.idle_unload_s):
            await self.unload(reason=f"idle for {settings.idle_unload_s:.0f}s")

    def status(self) -> dict:
        idle = time.monotonic() - self.last_used
        return {
            "state": self.state if self.alive or self.state == "loading" else "unloaded",
            "pid": self.proc.pid if self.alive else None,
            "idle_s": round(idle, 1) if self.alive else None,
            "unloads_in_s": round(max(0.0, settings.idle_unload_s - idle), 1)
            if self.alive and self.state == "ready" else None,
            **({"info": self.info} if self.info else {}),
        }

    # ---- protocol --------------------------------------------------------------------------

    async def _read_event(self) -> dict:
        line = await self.proc.stdout.readline()
        if not line:
            code = await self.proc.wait()
            raise EngineError(f"{self.name}: worker exited (code {code}); see `localtts logs`")
        return json.loads(line)

    async def generate(self, **params) -> AsyncIterator[bytes]:
        """Yield PCM s16le chunks; the final item is the worker's `done` stats dict."""
        async with self._lock:
            if self._stale and self.alive:
                await self._drain()
            await self.ensure_loaded()
            self.state = "busy"
            error = None
            try:
                self.proc.stdin.write((json.dumps({"op": "generate", **params}) + "\n").encode())
                await self.proc.stdin.drain()
                while True:
                    msg = await self._read_event()
                    ev = msg.get("event")
                    if ev == "chunk":
                        yield await self.proc.stdout.readexactly(msg["nbytes"])
                    elif ev == "done":
                        yield msg
                        return
                    elif ev == "error":
                        # The worker reported and is still in sync; keep it loaded.
                        error = EngineError(f"{self.name}: {msg.get('message')}")
                        break
            except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as exc:
                await self.unload(reason="worker died")
                raise EngineError(f"{self.name}: worker died mid-request ({exc})") from exc
            except BaseException:
                # Client went away mid-stream (e.g. Cancel in the UI): the worker is still
                # emitting audio for this request. Read it off before the next request rather
                # than restarting the worker, which would cost a full model load.
                if self.alive:
                    self._stale = True
                raise
            finally:
                if self.alive:
                    self.state = "ready"
                self.last_used = time.monotonic()
            raise error


    async def _drain(self) -> None:
        """Read off an abandoned request's remaining output; restart the worker if it takes long."""
        log.info("%s: finishing an abandoned request first", self.name)
        try:
            async with asyncio.timeout(DRAIN_TIMEOUT_S):
                while True:
                    msg = await self._read_event()
                    if msg.get("event") == "chunk":
                        await self.proc.stdout.readexactly(msg["nbytes"])
                    elif msg.get("event") in ("done", "error"):
                        break
            self._stale = False
        except (TimeoutError, asyncio.IncompleteReadError, EngineError, ValueError):
            await self.unload(reason=f"abandoned request still running after {DRAIN_TIMEOUT_S:.0f}s")


async def _oneshot(script: str, args: list[str], what: str, timeout: float = 600) -> dict:
    """Run a worker once in a throwaway process (Breeze venv, which has Whisper); return its result."""
    if not settings.breeze_python.exists():
        raise EngineError(f"{what} needs Whisper from the Breeze environment ({settings.breeze_python} not found)")
    proc = await asyncio.create_subprocess_exec(
        str(settings.breeze_python), str(WORKERS / script), *args,
        stdout=asyncio.subprocess.PIPE, stderr=None, cwd=str(WORKERS),
        env={**os.environ, "HF_HUB_OFFLINE": "1", **breeze_env()},
    )
    out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    for line in out.decode().splitlines():
        msg = json.loads(line)
        if msg.get("event") == "result":
            return msg
        if msg.get("event") == "error":
            raise EngineError(f"{what} failed: {msg.get('message')}")
    raise EngineError(f"{what} failed (exit {proc.returncode}); see `localtts logs`")


async def transcribe(path: Path, language: str | None = None) -> str:
    """Whisper transcript of a voice reference."""
    return (await _oneshot("transcribe_worker.py", [str(path)] + ([language] if language else []), "transcription"))["text"]


async def align_clips(job_file: Path) -> dict:
    """Word timings for finished clips: {id: {duration, sample_rate, words}}."""
    return (await _oneshot("align_worker.py", [str(job_file)], "alignment"))["items"]


ENGINES: dict[str, Engine] = {
    "breeze": Engine("breeze", settings.breeze_python, "breeze_worker.py",
                     env={"HF_HUB_OFFLINE": "1", **breeze_env(),
                          # Persistent torch.compile cache (the default lives in /tmp, wiped at boot).
                          "TORCHINDUCTOR_CACHE_DIR": str(Path("~/.cache/localtts/inductor").expanduser())}),
    "kokoro": Engine("kokoro", settings.kokoro_python, "kokoro_worker.py"),
}
