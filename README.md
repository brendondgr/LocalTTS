# LocalTTS

An always-on local text-to-speech API on **http://127.0.0.1:5040**, backed by
**Breeze TTS 2** (voice cloning, voice design, voice direction) and **Kokoro-82M**
(fast built-in voices). It runs in the background as a systemd user service, starts at
boot, and keeps nothing on the GPU while unused: each engine is unloaded after
**10 minutes** without a request, and loads again on the next one.

**Web UI: http://127.0.0.1:5040** (`localtts ui` opens it) · API docs: http://127.0.0.1:5040/docs

> **Licence.** Breeze TTS 2 weights and outputs are research / non-commercial only.
> Treat Breeze voices as test voices. Kokoro (Apache-2.0) has no such limit.

## Web UI

A chat: type text, pick a voice, get audio back. Everything the API does is in it:

- **Voices** (sidebar): saved voices with their reference clip and transcript (editable), plus
  **Add voice** (upload, drag and drop, or record from the microphone; Whisper writes the
  transcript if you leave it empty) and **Design voice** (describe one, Breeze keeps a sample).
- **Composer**: voice picker (your voices, "Describe a voice…", Kokoro voices), **Direction**
  (emotion, tone, pace for Breeze), **Tags** (insert `(sigh)`, `(laugh)`… at the cursor) and
  **Enhance** (below). The sliders button holds saving on/off, save folder, format, seed,
  guidance, Kokoro speed and the Enhance style.
- Each reply has a player, download, regenerate (new seed), edit-and-resend and copy.
- **Settings**: the backend (this machine or a remote one, below), engine load/unload, the
  Enhance model, and clearing history.

Chats are kept in the browser (IndexedDB). With saving off, the audio is held in memory only
and is gone after a reload, so nothing is written anywhere.

### AI-Enhance

**Enhance** sends the draft to an LLM, which inserts Breeze vocal-event tags where they fit
and writes a delivery direction (e.g. "Resigned but lightly amused, unhurried."). The draft is
replaced in place so you can review it, with **Undo**. It never changes your words; if the model
does, the reply is flagged. *Subtle* adds about one tag per two or three sentences, *Expressive*
more. Kokoro voices can't perform tags, so Enhance is off for them and tags are stripped from
Kokoro requests.

Tags Breeze documents (the only ones Enhance uses): `(laugh)` `(sigh)` `(cough)` `(clears throat)`.
The Tags menu also lists experimental ones (`(chuckle)` `(gasp)` `(groan)` `(sniff)` `(breath)`
`(hmm)` `(giggle)` `(cry)`): Breeze does not read them out as words, but whether each makes the
sound has not been checked by ear.

The LLM is any OpenAI-compatible endpoint: by default the DashLLM relay (`http://127.0.0.1:4000/v1`,
model `auto`), which routes to whatever model is running. If none is, start one:

```bash
lcpp start gemma4-E4B      # ~10 GB; about 1.5 s per Enhance
```

`POST /v1/enhance {"text", "style": "subtle|expressive", "model"?, "direction"?}` returns
`{"text", "delivery", "tags_added", "model", "warning"?}`; `GET /v1/enhance/tags` lists the tags.

### Remote GPU backend

The UI can drive another LocalTTS, e.g. one on the CUDA machine. Run LocalTTS there as usual
(it stays bound to its own `127.0.0.1:5040`), forward it to this machine, and point the UI at it
in **Settings → Backend → Another LocalTTS**:

```bash
ssh -N -L 5041:127.0.0.1:5040 user@gpu-box
```

then use `http://localhost:5041`. Voices, engines and saved files are that machine's;
Enhance still runs here. The API accepts cross-origin requests from `localhost`/`127.0.0.1`
pages only, so other websites cannot drive it.

## Control

```
localtts start | stop | restart     # manage the background service
localtts status                     # service + engine state, idle timers
localtts logs [-f] [N]              # service log (journalctl)
localtts enable | disable           # start at boot (enabled by install.sh)
localtts load [breeze|kokoro]       # load now instead of on first request
localtts unload [engine]            # free the GPU now (default: all)
localtts voices                     # saved + built-in voices
localtts ui                         # open the web UI
```

Generation goes through the web UI or the API.

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
| `save_dir` | `~/Music/TTS` | folder to save into (on the server); relative paths are under `~/Music/TTS` |

```bash
# saved to ~/Music/TTS/ and returned
curl http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "Hello there.", "name": "Brendon"}' -o hello.wav

# saved to ~/Music/TTS/podcast/ (or give an absolute path)
curl http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "(sigh) Episode two.", "name": "Brendon", "save_dir": "podcast"}' -o ep2.wav

# returned only, never written on this machine
curl http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "Hello there.", "name": "Brendon", "no_save": true, "format": "mp3"}' -o hello.mp3

# stream and play as it arrives
curl -N http://127.0.0.1:5040/v1/speech -H 'content-type: application/json' \
     -d '{"text": "Streaming now.", "name": "af_heart", "stream": true, "no_save": true}' \
  | aplay -q -f S16_LE -r 24000 -c 1
```

Response headers: `X-LocalTTS-Engine`, `X-LocalTTS-Voice`, `X-Audio-Duration`, `X-RTF`,
`X-LocalTTS-Saved` (the saved file name, or `no`), `X-LocalTTS-Path` (percent-encoded).

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

Everything generated without `no_save` lands in `~/Music/TTS/` (change it with
`LOCALTTS_OUTPUTS_DIR`), or in the request's `save_dir`. The endpoints below cover the
default folder.

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

Saved voices live in `data/voices/<name>/{reference.wav,voice.json}` (git-ignored).
Generated audio goes to `~/Music/TTS/` (`LOCALTTS_OUTPUTS_DIR`) unless `no_save` is set.

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
| `LOCALTTS_OUTPUTS_DIR` | `~/Music/TTS` | where generated audio is saved (unless `no_save`) |
| `LOCALTTS_PRELOAD` | empty | e.g. `breeze` to load at service start |
| `LOCALTTS_DEFAULT_ENGINE` | kokoro | engine for requests with no voice and no instruction |
| `LOCALTTS_LLM_URL` | `http://127.0.0.1:4000/v1` | OpenAI-compatible endpoint for AI-Enhance |
| `LOCALTTS_LLM_MODEL` | `auto` | model id sent to it (the UI can pick another) |
| `LOCALTTS_BREEZE_FAST` | `depth_decoder,backbone_decode` | empty = eager (RTF ~2.7, no warmup) |

The full list is in `localtts.env.example`.

## Troubleshooting

- `localtts logs` shows the server and both workers (worker lines start with `[worker]`).
- A failed engine load reports `worker exited (code N)`; the traceback is in the log.
- Breeze and the vLLM/llama.cpp services share the 128 GB unified memory; if a load fails
  for memory, `localtts unload` or stop the other service.
