"""Thin wrapper around a local whisper.cpp server + pseudo-streaming transcription.

whisper.cpp has no true streaming API over HTTP, so we do utterance-based
pseudo-streaming: buffer mic audio, cut a segment when the speaker pauses
(or the segment gets too long), and transcribe each segment as it closes.
"""
from __future__ import annotations

import io
import os
import queue
import subprocess
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import requests

from agent import jarvis

# ── whisper.cpp locations (override via env / .env) ──────────────────────────
WHISPER_CPP_DIR = Path(
    os.getenv("WHISPER_CPP_DIR", Path(__file__).resolve().parents[2] / "whispercpp" / "whisper.cpp")
)
WHISPER_MODEL = Path(
    os.getenv("WHISPER_MODEL", WHISPER_CPP_DIR / "models" / "ggml-base.en.bin")
)
SERVER_EXE = WHISPER_CPP_DIR / "build" / "bin" / "Release" / "whisper-server.exe"

# ── Segmentation tuning ───────────────────────────────────────────────────────
MIN_SPEECH_RMS = 150       # ponytail: energy VAD with adaptive noise floor; swap in Silero VAD if it misfires
SILENCE_HANG_MS = 800      # this much trailing silence closes an utterance
MAX_SEGMENT_S = 15         # hard cap so a monologue still produces output
PREROLL_S = 0.5            # audio kept before speech starts, so first word isn't clipped
DEBUG = os.getenv("STT_DEBUG", "") not in ("", "0")  # set STT_DEBUG=1 to print per-chunk levels


class WhisperCpp:
    """Starts whisper-server.exe (model loaded once) and transcribes PCM chunks over HTTP."""

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
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        buf.seek(0)

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


def stream_transcribe(
    audio_q: queue.Queue[bytes | None],
    whisper: WhisperCpp,
    sample_rate: int,
    on_text: Callable[[str], None] = lambda t: print(f"[final] {t}"),
) -> None:
    """
    Runs in its own thread. Reads raw int16 PCM chunks from *audio_q*,
    segments them on silence, and transcribes each utterance with whisper.cpp.
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
                text = whisper.transcribe(bytes(buf), sample_rate)
                if text:
                    on_text(text)
                    print(jarvis.invoke({
                        "messages": [{"role": "user", "content": text}]
                    }))
            except requests.RequestException as exc:
                print(f"[whisper error: {exc}]")
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
