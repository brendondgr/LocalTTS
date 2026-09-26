# LocalTTS

An always-on local text-to-speech API on **http://127.0.0.1:5040**, backed by
**Breeze TTS 2** (voice cloning, voice design, voice direction) and **Kokoro-82M**
(fast built-in voices). It runs in the background as a systemd user service, starts at
boot, and keeps nothing on the GPU while unused: each engine is unloaded after
**10 minutes** without a request, and loads again on the next one.

Interactive docs: http://127.0.0.1:5040/docs

> **Licence.** Breeze TTS 2 weights and outputs are research / non-commercial only.
> Treat Breeze voices as test voices. Kokoro (Apache-2.0) has no such limit.

## Control

```
localtts start | stop | restart     # manage the background service
localtts status                     # service + engine state, idle timers
localtts logs [-f] [N]              # service log (journalctl)
localtts enable | disable           # start at boot (enabled by install.sh)
localtts load [breeze|kokoro]       # load now instead of on first request
localtts unload [engine]            # free the GPU now (default: all)
localtts voices                     # saved + built-in voices
```

Generation goes through the API only.

## Voices

| Kind | How to call it | Engine |
|---|---|---|
| Saved voice (your upload) | `"name": "Brendon"` | Breeze clone |
| Saved voice + delivery | `"name": "Brendon", "instruction": "whisper, nervous"` | Breeze direction |
| Designed on the fly | no name, `"instruction": "a gravelly old sailor"` | Breeze design (voice varies per call) |
| Kokoro built-in | `"name": "af_heart"` (see `localtts voices`) | Kokoro |
| Nothing given | default Kokoro voice (`af_heart`) | Kokoro |

Names are case-insensitive. A designed voice changes with every text; to keep one, save
it with `POST /v1/voices/design`, which stores a sample as a normal cloneable voice.

### Add a voice

```bash
# with the exact transcript (best)
curl -F name=Brendon -F audio=@me.wav -F "transcript=Exactly what I say in the clip." \
     http://127.0.0.1:5040/v1/voices

# without one: Whisper writes it (check it afterwards; clone quality depends on it)
curl -F name=Brendon -F audio=@me.m4a http://127.0.0.1:5040/v1/voices
```

Any format ffmpeg reads; 3 to 60 s accepted, **10 to 25 s of clean speech** is the sweet
spot. Add `-F overwrite=true` to replace an existing voice.

```bash
curl http://127.0.0.1:5040/v1/voices                         # list
curl http://127.0.0.1:5040/v1/voices/Brendon/audio -o ref.wav # the stored reference
curl -X PATCH http://127.0.0.1:5040/v1/voices/Brendon -H 'content-type: application/json' \
     -d '{"transcript": "Corrected transcript."}'            # or {"rename": "New Name"}
curl -X DELETE http://127.0.0.1:5040/v1/voices/Brendon        # permanent (no trash)

# invent a voice and keep it
curl http://127.0.0.1:5040/v1/voices/design -H 'content-type: application/json' \
     -d '{"name": "Narrator", "instruction": "A calm, deep documentary narrator."}'
```

## Generate

`POST /v1/speech` with JSON:

| Field | Default | |
|---|---|---|
| `text` | required | anything up to 20k characters; long text is split at sentences for Breeze |
| `name` (or `voice`) | none | saved voice or Kokoro voice, see above |
| `engine` | `auto` | `breeze` / `kokoro` to force one |
| `instruction` | none | Breeze: voice description (design) or delivery (direction). Tags like `(sigh)` `(laughs)` also work inside `text` |
| `cfg_scale` | 1, or 4 with an instruction | Breeze guidance strength |
| `seed` | 42 | Breeze sampling seed |
| `speed` | 1.0 | Kokoro only |
| `format` | `wav` | `wav` `flac` `mp3` `ogg` `pcm` (raw s16le, 24 kHz mono) |
| `stream` | false | stream raw PCM s16le as it is generated |
| `no_save` | false | **return the audio only; nothing is written to disk** |

```bash
# saved to data/outputs/ and returned
curl http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "Hello there.", "name": "Brendon"}' -o hello.wav

# returned only, never written on this machine
curl http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "Hello there.", "name": "Brendon", "no_save": true, "format": "mp3"}' -o hello.mp3

# stream and play as it arrives
curl -N http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "Streaming now.", "name": "af_heart", "stream": true, "no_save": true}' \
  | aplay -q -f S16_LE -r 24000 -c 1
```

Response headers: `X-LocalTTS-Engine`, `X-LocalTTS-Voice`, `X-Audio-Duration`, `X-RTF`,
`X-LocalTTS-Saved` (the saved file name, or `no`), `X-LocalTTS-Path`.

Python:

```python
import requests
r = requests.post("http://127.0.0.1:5040/v1/speech",
                  json={"text": "Hello there.", "name": "Brendon", "no_save": True})
r.raise_for_status()
open("hello.wav", "wb").write(r.content)
```

**OpenAI-compatible**: `POST /v1/audio/speech` takes `{model, input, voice, instructions,
response_format, speed}` (plus `no_save`), so OpenAI TTS clients work by pointing their
base URL at `http://127.0.0.1:5040/v1`. `model` may be `breeze`, `kokoro` or anything else (auto).

### Saved outputs

```bash
curl http://127.0.0.1:5040/v1/outputs                     # list (newest first)
curl http://127.0.0.1:5040/v1/outputs/<file> -o out.wav   # fetch
curl -X DELETE http://127.0.0.1:5040/v1/outputs/<file>    # permanent (no trash)
```

## Performance (Strix Halo, gfx1151)

| | Breeze | Kokoro |
|---|---|---|
| cold start (first request after unload) | ~30 s (load + fast-path warmup; the first ever start also compiles, ~50 s) | ~2 s |
| warm speed | RTF ~1.5 (1 s of audio takes ~1.5 s) | RTF ~0.5 |
| GPU memory while loaded | ~8.5 GiB | small (82M-parameter model) |

One request at a time per engine; Breeze and Kokoro can run side by side.

## How it works

```
client ──HTTP──> localtts (FastAPI, .venv)  ──stdin/stdout──> breeze_worker.py  (~/venvs/breeze-next)
                                           └─stdin/stdout──> kokoro_worker.py  (~/venvs/kokoro)
```

- Each engine is a subprocess in **its own venv**; the server never installs into them.
  Audio comes back over a pipe (JSON header + PCM bytes), so `no_save` really touches no disk.
- **Unload = stop the worker process**, which returns every byte of GPU memory. A reaper
  checks every 15 s and stops any engine idle for `LOCALTTS_IDLE_UNLOAD_S` (600).
- Breeze runs with the `depth_decoder` + `backbone_decode` fast stages (the fastest
  combination on this GPU that also handles long references). The torch.compile cache
  lives in `~/.cache/localtts/inductor` so it survives reboots.
- Voice uploads are converted by ffmpeg to 24 kHz mono WAV; missing transcripts come from
  a one-shot Whisper large-v3-turbo process that exits right after.
- Deletes use `unlink`/`rmtree`: permanent, never the desktop Trash.

Data lives in `data/` (git-ignored): `data/voices/<name>/{reference.wav,voice.json}` and
`data/outputs/`.

## Install / reinstall

```bash
./install.sh
```

Creates `.venv`, installs `~/.config/systemd/user/localtts.service` (enabled, with linger
so it starts at boot without a login), copies `localtts.env.example` to
`~/.config/localtts/localtts.env`, and links `localtts` into `~/.local/bin`. Idempotent.

Requirements: `uv`, `ffmpeg`, the Breeze checkout (`~/Projects/breeze-tts`, branch
`rocm-port`) with weights in `~/models/breeze-tts-2` and its venv `~/venvs/breeze-next`,
and the Kokoro venv `~/venvs/kokoro`.

## Configuration

Edit `~/.config/localtts/localtts.env`, then `localtts restart`. Common ones:

| Variable | Default | |
|---|---|---|
| `LOCALTTS_PORT` | 5040 | |
| `LOCALTTS_HOST` | 127.0.0.1 | `0.0.0.0` exposes it to the network; there is **no auth** |
| `LOCALTTS_IDLE_UNLOAD_S` | 600 | idle seconds before an engine leaves the GPU |
| `LOCALTTS_PRELOAD` | empty | e.g. `breeze` to load at service start |
| `LOCALTTS_DEFAULT_ENGINE` | kokoro | engine for requests with no voice and no instruction |
| `LOCALTTS_BREEZE_FAST` | `depth_decoder,backbone_decode` | empty = eager (RTF ~2.7, no warmup) |

The full list is in `localtts.env.example`.

## Troubleshooting

- `localtts logs` shows the server and both workers (worker lines start with `[worker]`).
- A failed engine load reports `worker exited (code N)`; the traceback is in the log.
- Breeze and the vLLM/llama.cpp services share the 128 GB unified memory; if a load fails
  for memory, `localtts unload` or stop the other service.
