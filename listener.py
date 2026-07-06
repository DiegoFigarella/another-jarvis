from __future__ import annotations

import os
import queue
import threading
from enum import Enum, auto

import numpy as np
import pyaudio
import pygame
from clapDetector import ClapDetector
from dotenv import load_dotenv

load_dotenv()

from whisper_stt import WhisperCpp, stream_transcribe  # noqa: E402  (needs .env loaded first)

# ── Audio config ──────────────────────────────────────────────────────────────
THRESHOLD_BIAS: int = 8000
LOWCUT: int = 1000
HIGHCUT: int = 10000
CHUNK_FRAMES: int = 1024          # frames per audio buffer read
STT_CHUNK_MS: int = 500           # ms of audio per chunk handed to the STT thread
# Device, channel count, and sample rate are taken from the system default mic
# at startup, so plugging in earbuds just works.


class State(Enum):
    IDLE = auto()
    LISTENING = auto()


# ── Device selection ──────────────────────────────────────────────────────────

def pick_input_device() -> int:
    """Return PyAudio's default input device, or the first device that can record.

    PyAudio sometimes reports no default input on Windows even when mics exist
    under other host APIs (WASAPI etc.), so we scan everything ourselves.
    """
    p = pyaudio.PyAudio()
    try:
        candidates: list[int] = []
        print("Available input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                candidates.append(i)
                print(f"  {i}: {info['name']} "
                      f"({info['maxInputChannels']} ch @ {int(info['defaultSampleRate'])} Hz)")
        try:
            return p.get_default_input_device_info()["index"]
        except OSError:
            if candidates:
                return candidates[0]
            raise RuntimeError(
                "No microphone found - connect/enable one in Windows sound settings"
            ) from None
    finally:
        p.terminate()


# ── Clap detection ────────────────────────────────────────────────────────────

def detect_double_clap(detector: ClapDetector, audio: np.ndarray) -> bool:
    result = detector.run(
        thresholdBias=THRESHOLD_BIAS,
        lowcut=LOWCUT,
        # clamp below Nyquist: bluetooth mics run at 8-16 kHz and scipy's
        # bandpass filter throws if highcut >= rate/2
        highcut=min(HIGHCUT, int(detector.rate * 0.45)),
        audioData=audio,
    )
    return len(result) == 2


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.mixer.init()

    # INPUT_DEVICE env var (index or name substring) overrides auto-detection
    device_env = os.getenv("INPUT_DEVICE", "").strip()
    if device_env:
        input_device = int(device_env) if device_env.lstrip("-").isdigit() else device_env
    else:
        input_device = pick_input_device()

    whisper = WhisperCpp()
    print("[whisper.cpp server ready]")

    detector = ClapDetector(
        inputDevice=input_device,
        logLevel=10,
        bufferLength=CHUNK_FRAMES,
        debounceTimeFactor=0.15,
        rate=None,                    # use the device's native rate
    )
    detector.initAudio()
    sample_rate: int = detector.rate
    channels: int = detector.p.get_device_info_by_index(detector.inputDevice)["maxInputChannels"]
    stt_chunk_bytes: int = int(sample_rate * STT_CHUNK_MS / 1000) * 2  # int16 mono

    state = State.IDLE

    # Queue shared between the sounddevice callback and the STT thread.
    # bytes  → audio chunk to transcribe
    # None   → sentinel: shut down STT thread
    audio_q: queue.Queue[bytes | None] = queue.Queue()
    stt_thread: threading.Thread | None = None

    # Accumulate frames so we send STT_CHUNK_FRAMES at a time (smoother stream)
    pcm_buffer: bytearray = bytearray()

    def flush_to_stt(buf: bytearray, force: bool = False) -> bytearray:
        """Send complete STT-chunk-sized slices; return leftover bytes."""
        while len(buf) >= stt_chunk_bytes or (force and buf):
            audio_q.put(bytes(buf[:stt_chunk_bytes]))
            buf = buf[stt_chunk_bytes:]
        return buf

    try:
        while True:
            # ClapDetector.getAudio() is a blocking call that yields one buffer
            raw: np.ndarray = detector.getAudio()

            # ── Clap detection ─────────────────────────────────────────────
            if detect_double_clap(detector, raw):

                if state is State.LISTENING:
                    # Second double-clap → stop
                    print("\n[Stopping assistant]")
                    pcm_buffer = flush_to_stt(pcm_buffer, force=True)
                    audio_q.put(None)          # signal STT thread to exit
                    if stt_thread is not None:
                        stt_thread.join()
                    pcm_buffer = bytearray()
                    state = State.IDLE
                    break

                # First double-clap → wake
                print("[Wake detected]")
                pygame.mixer.music.load("should_i_stay_or_should_i_go.mp3")
                pygame.mixer.music.play()

                audio_q = queue.Queue()        # fresh queue for this session
                pcm_buffer = bytearray()
                stt_thread = threading.Thread(
                    target=stream_transcribe,
                    args=(audio_q, whisper, sample_rate),
                    daemon=True,
                )
                stt_thread.start()
                state = State.LISTENING
                continue

            # ── Forward audio to STT while listening ───────────────────────
            # ponytail: wake song and STT run concurrently. Fine on headphones;
            # on speakers the mic may transcribe the song's lyrics.
            if state is State.LISTENING:
                # downmix interleaved multi-channel capture to mono for whisper
                mono = raw if channels == 1 else raw.reshape(-1, channels).mean(axis=1).astype(np.int16)
                pcm_buffer.extend(mono.tobytes())
                pcm_buffer = flush_to_stt(pcm_buffer)

    except KeyboardInterrupt:
        print("\n[Exited gracefully]")

    finally:
        if state is State.LISTENING:
            audio_q.put(None)
            if stt_thread is not None:
                stt_thread.join()
        whisper.close()
        detector.stop()


if __name__ == "__main__":
    main()