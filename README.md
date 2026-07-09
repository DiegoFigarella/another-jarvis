# another-jarvis

A hands-free WhatsApp assistant. Double-clap to wake it, speak, and double-clap
again to finish the session. Audio can be transcribed by either a local
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) server or OpenAI's hosted
Whisper API.

## How it works

```text
microphone -> listener.py -> double clap -> utterance segmentation
                                      -> whisper.cpp or OpenAI Whisper
                                      -> LangChain agent -> Evolution API
```

Transcription is utterance-based: the listener sends a segment after about 0.8
seconds of silence or after 15 seconds of continuous speech.

## Setup

### 1. Install Python dependencies

Create a virtual environment, then install the requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. On macOS, if PyAudio
cannot build, install PortAudio first with `brew install portaudio` and retry.

### 2. Configure the environment

```bash
cp .env.example .env
```

Fill in the Evolution API values and the LangChain model provider key. The
included example uses an OpenAI model, so `OPENAI_API_KEY` covers both the agent
and OpenAI transcription.

### 3. Choose a transcription backend

For hosted OpenAI Whisper, set:

```dotenv
STT_BACKEND=openai
OPENAI_API_KEY=your-key
OPENAI_TRANSCRIPTION_MODEL=whisper-1
```

For fully local transcription, set `STT_BACKEND=whisper_cpp`, clone
whisper.cpp, and build the server:

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target whisper-server
```

Download a model from the whisper.cpp repository, then configure its paths:

```dotenv
STT_BACKEND=whisper_cpp
WHISPER_CPP_DIR=/absolute/path/to/whisper.cpp
WHISPER_MODEL=/absolute/path/to/whisper.cpp/models/ggml-base.en.bin
```

The server executable is detected in the usual CMake output directories on
macOS, Linux, and Windows. Set `WHISPER_SERVER` only for a custom location.

## macOS microphone setup

Grant microphone access to the terminal or app running Jarvis under **System
Settings > Privacy & Security > Microphone**.

The listener uses the system default microphone. To override it, set
`INPUT_DEVICE` to either an index or a case-insensitive name substring:

```dotenv
INPUT_DEVICE=0
# or
INPUT_DEVICE=MacBook Pro Microphone
```

Available input devices are printed at startup. Speaker-only devices are
excluded and invalid selections fail with a specific error.

## Run

```bash
python listener.py
```

- First double-clap: start listening.
- Speak normally; pauses close individual utterances.
- Second double-clap: flush the final utterance and exit.

The wake sound and transcription run concurrently. Headphones avoid the wake
sound being picked up by the microphone.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `STT_BACKEND` | `whisper_cpp` | `whisper_cpp` or `openai` |
| `INPUT_DEVICE` | system default | Input index or name substring |
| `OPENAI_TRANSCRIPTION_MODEL` | `whisper-1` | OpenAI transcription model |
| `OPENAI_TRANSCRIPTION_PROMPT` | WhatsApp phrase hints | Optional vocabulary hint |
| `WHISPER_CPP_DIR` | `~/whispercpp/whisper.cpp` | Local whisper.cpp checkout |
| `WHISPER_MODEL` | `models/ggml-base.en.bin` | Local model path |
| `WHISPER_SERVER` | auto-detected | Local server executable |
| `STT_DEBUG` | off | Set to `1` for per-chunk audio levels |

## Tests

The unit tests use fake audio and API clients, so they do not need a microphone,
OpenAI request, or running Evolution API:

```bash
python -m unittest
```

To smoke-test the local whisper.cpp server with its bundled `jfk.wav` sample:

```bash
STT_BACKEND=whisper_cpp python whisper_stt.py
```
