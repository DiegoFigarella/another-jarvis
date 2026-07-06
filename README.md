# another-jarvis

Voice assistant: double-clap to wake, speak, and your words are transcribed
locally with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (no cloud
STT). A second double-clap stops the session.

## How it works

```
mic ─▶ listener.py ─▶ double clap? ─▶ start session
              │
              ▶ audio chunks ─▶ whisper_stt.stream_transcribe (thread)
                                      │  buffers audio, cuts a segment when
                                      │  you pause (~0.8 s) or after 15 s
                                      ▼
                               whisper-server.exe (local HTTP, model loaded once)
                                      │
                                      ▼
                               [final] transcript printed per utterance
```

whisper.cpp has no true streaming API, so this is utterance-based
pseudo-streaming: you get a transcript a moment after each pause instead of
word-by-word interim results. Latency per utterance ≈ pause detection (~1 s)
plus inference (well under a second for `base.en` on CPU).

## Setup

### 1. Build whisper.cpp (one-time)

The whisper.cpp clone is expected at `..\..\whispercpp\whisper.cpp` relative
to this folder (override with the `WHISPER_CPP_DIR` env var). Build the server:

```powershell
cd ..\..\whispercpp\whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target whisper-server
```

> **Antivirus note:** AVG/Avast tend to flag the freshly built
> `whisper-server.exe` as `IDP.Generic` — a false positive on new unsigned
> binaries. If that happens, restore it from quarantine and add an exclusion
> for the `whispercpp` folder, then rebuild.

### 2. Model

`models/ggml-base.en.bin` is already in the whisper.cpp clone. To use another
model, download it with `models\download-ggml-model.cmd <name>` and set
`WHISPER_MODEL` to its path.

### 3. Python deps

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python listener.py
```

- **Double clap** → wake sound plays, transcription session starts.
- Speak; each time you pause, a `[final] ...` line is printed.
- **Double clap again** → session stops and the program exits.

## Configuration

| What | Where | Default |
|------|-------|---------|
| whisper.cpp repo path | `WHISPER_CPP_DIR` env var | `..\..\whispercpp\whisper.cpp` |
| Model file | `WHISPER_MODEL` env var | `models/ggml-base.en.bin` in the repo |
| Speech threshold (mic sensitivity) | `MIN_SPEECH_RMS` in [whisper_stt.py](whisper_stt.py); adapts to ambient noise on top of this floor | 150 |
| Per-chunk level debugging | `STT_DEBUG=1` env var | off |
| Pause length that ends an utterance | `SILENCE_HANG_MS` in [whisper_stt.py](whisper_stt.py) | 800 ms |
| Input device | `INPUT_DEVICE` env var: device index (e.g. `5`) or name substring (e.g. `Buds2`) | empty → auto (system default mic) |
| Sample rate | auto: the chosen device's native rate | — |

## Smoke test

Transcribes whisper.cpp's bundled `jfk.wav` through the full
server-start → HTTP → transcript pipeline:

```powershell
python whisper_stt.py
```

Prints the JFK quote and `OK` on success.
