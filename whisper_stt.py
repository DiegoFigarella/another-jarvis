"""Speech-to-text backends and pseudo-streaming transcription.

The listener buffers microphone audio, cuts a segment when the speaker pauses
(or the segment gets too long), and sends each completed utterance to the
configured backend. Both whisper.cpp and OpenAI use this same segmentation.
"""
from __future__ import annotations

import io
import os
import queue
import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# ── whisper.cpp locations (override via env / .env) ──────────────────────────


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else default


WHISPER_CPP_DIR = _env_path(
    "WHISPER_CPP_DIR",
    Path.home() / "whispercpp" / "whisper.cpp",
)
WHISPER_MODEL = _env_path(
    "WHISPER_MODEL",
    WHISPER_CPP_DIR / "models" / "ggml-base.en.bin",
)


def _default_server_executable() -> Path:
    override = os.getenv("WHISPER_SERVER", "").strip()
    if override:
        return Path(override).expanduser()

    executable = "whisper-server.exe" if os.name == "nt" else "whisper-server"
    candidates = (
        WHISPER_CPP_DIR / "build" / "bin" / "Release" / executable,
        WHISPER_CPP_DIR / "build" / "bin" / executable,
        WHISPER_CPP_DIR / "build" / "bin" / "Debug" / executable,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


SERVER_EXE = _default_server_executable()

# ── Segmentation tuning ───────────────────────────────────────────────────────
MIN_SPEECH_RMS = 150       # ponytail: energy VAD with adaptive noise floor; swap in Silero VAD if it misfires
SILENCE_HANG_MS = 800      # this much trailing silence closes an utterance
MAX_SEGMENT_S = 15         # hard cap so a monologue still produces output
PREROLL_S = 0.5            # audio kept before speech starts, so first word isn't clipped
DEBUG = os.getenv("STT_DEBUG", "") not in ("", "0")  # set STT_DEBUG=1 to print per-chunk levels


class Transcriber(Protocol):
    """Minimal interface shared by local and hosted transcription backends."""

    def transcribe(self, pcm: bytes, sample_rate: int) -> str: ...

    def close(self) -> None: ...


def _wav_buffer(pcm: bytes, sample_rate: int) -> io.BytesIO:
    if not pcm:
        raise ValueError("No audio was captured")
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    buf.seek(0)
    buf.name = "audio.wav"
    return buf


class WhisperCpp:
    """Start whisper.cpp's HTTP server and transcribe PCM chunks locally."""

    def __init__(
        self,
        model: Path = WHISPER_MODEL,
        server_exe: Path = SERVER_EXE,
        port: int = 8910,
    ) -> None:
        if not server_exe.exists():
            raise FileNotFoundError(
                f"{server_exe} not found - build whisper.cpp first (see README)"
            )
        if not model.exists():
            raise FileNotFoundError(f"model not found: {model}")

        self._url = f"http://127.0.0.1:{port}/inference"
        self._proc = subprocess.Popen(
            [str(server_exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_ready(port)

    def _wait_ready(self, port: int, timeout_s: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("whisper-server exited during startup")
            try:
                requests.get(f"http://127.0.0.1:{port}/", timeout=2)
                return
            except requests.ConnectionError:
                time.sleep(0.25)
            except requests.Timeout:
                return  # port accepted the connection → server is up, just slow to reply
        self.close()
        raise TimeoutError("whisper-server did not become ready in time")

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe raw mono int16 PCM. The server resamples, so any rate works."""
        buf = _wav_buffer(pcm, sample_rate)

        response = requests.post(
            self._url,
            files={"file": ("audio.wav", buf, "audio/wav")},
            data={"response_format": "json", "temperature": "0.0"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class OpenAIWhisper:
    """Transcribe utterances with OpenAI's audio transcription API."""

    def __init__(
        self,
        model: str | None = None,
        prompt: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "The OpenAI backend requires the 'openai' package. "
                    "Run: pip install -r requirements.txt"
                ) from exc
            client = OpenAI()
            self._owns_client = True
        else:
            self._owns_client = False

        self._client = client
        self._model = model or os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1")
        self._prompt = (
            prompt
            if prompt is not None
            else os.getenv("OPENAI_TRANSCRIPTION_PROMPT", "WhatsApp. Send a text message to.")
        )

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        audio_file = _wav_buffer(pcm, sample_rate)
        request: dict[str, Any] = {
            "model": self._model,
            "file": audio_file,
            "language": "auto",
        }
        if self._prompt:
            request["prompt"] = self._prompt

        result = self._client.audio.transcriptions.create(**request)
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return text.strip()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def create_transcriber(backend: str | None = None) -> Transcriber:
    """Create the backend selected by STT_BACKEND."""
    configured = backend if backend is not None else os.getenv("STT_BACKEND")
    selected = (configured or "whisper_cpp").strip().lower()
    selected = selected.replace("-", "_").replace(".", "_")
    if selected in {"whisper_cpp", "local"}:
        return WhisperCpp()
    if selected in {"openai", "openai_whisper"}:
        return OpenAIWhisper()
    raise ValueError(
        f"Unknown STT_BACKEND {selected!r}; expected 'whisper_cpp' or 'openai'"
    )


def stream_transcribe(
    audio_q: queue.Queue[bytes | None],
    transcriber: Transcriber,
    sample_rate: int,
    on_text: Callable[[str], None] = lambda t: print(f"[final] {t}"),
) -> None:
    """
    Runs in its own thread. Reads raw int16 PCM chunks from *audio_q*,
    segments them on silence, and transcribes each utterance with the selected backend.
    Exits when it receives the None sentinel from the queue.
    """
    buf = bytearray()
    silence_ms = 0.0
    in_speech = False
    noise_floor: float | None = None   # adapts to ambient level; speech = well above it
    preroll_bytes = int(PREROLL_S * sample_rate) * 2
    max_segment_bytes = MAX_SEGMENT_S * sample_rate * 2

    def flush() -> None:
        nonlocal buf, silence_ms, in_speech
        if in_speech:
            try:
                text = transcriber.transcribe(bytes(buf), sample_rate)
            except Exception as exc:  # noqa: BLE001 - backend errors should not kill audio capture
                print(f"[transcription error: {exc}]")
            else:
                try:
                    if text:
                        on_text(text)
                except Exception as exc:  # noqa: BLE001 - keep listening after agent errors
                    print(f"[assistant error: {exc}]")
        buf = bytearray()
        silence_ms = 0.0
        in_speech = False

    while True:
        chunk = audio_q.get()
        if chunk is None:
            flush()
            return

        buf.extend(chunk)
        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        noise_floor = rms if noise_floor is None else min(noise_floor, rms)
        threshold = max(MIN_SPEECH_RMS, 2.5 * noise_floor)
        loud = rms >= threshold
        if DEBUG:
            print(f"[rms {rms:5.0f}  thr {threshold:5.0f}  {'SPEECH' if loud else 'quiet'}]")
        if loud:
            in_speech = True
            silence_ms = 0.0
        else:
            noise_floor = 0.9 * noise_floor + 0.1 * rms  # drift with ambient noise
            silence_ms += 1000.0 * len(samples) / sample_rate

        if not in_speech:
            # waiting for speech: keep only a short pre-roll
            if len(buf) > preroll_bytes:
                del buf[: len(buf) - preroll_bytes]
        elif silence_ms >= SILENCE_HANG_MS or len(buf) >= max_segment_bytes:
            flush()


if __name__ == "__main__":
    # smoke test: transcribe the repo's sample recording through the whole pipeline
    sample = WHISPER_CPP_DIR / "samples" / "jfk.wav"
    with wave.open(str(sample), "rb") as wav:
        rate = wav.getframerate()
        pcm = wav.readframes(wav.getnframes())
    w = WhisperCpp()
    try:
        text = w.transcribe(pcm, rate)
        print(text)
        assert "country" in text.lower(), text
        print("OK")
    finally:
        w.close()
