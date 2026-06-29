from __future__ import annotations

import os
import queue
import threading
from enum import Enum, auto
from typing import Generator

import pygame
import sounddevice as sd
from clapDetector import ClapDetector, printDeviceInfo
from dotenv import load_dotenv
from google.cloud import speech

load_dotenv()

# ── Audio config ──────────────────────────────────────────────────────────────
THRESHOLD_BIAS: int = 8000
LOWCUT: int = 1000
HIGHCUT: int = 10000
SAMPLE_RATE: int = 48000          # you may adjust this to your microphone's sample rate
CHUNK_FRAMES: int = 1024          # frames per sounddevice callback
STT_CHUNK_MS: int = 500           # ms of audio per streaming-STT request chunk
STT_CHUNK_FRAMES: int = int(SAMPLE_RATE * STT_CHUNK_MS / 1000)


class State(Enum):
    IDLE = auto()
    LISTENING = auto()


# ── Google Streaming STT ──────────────────────────────────────────────────────

def _build_stt_client() -> speech.SpeechClient:
    return speech.SpeechClient()


def _make_streaming_config() -> speech.StreamingRecognitionConfig:
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="en-US", # feel free to change this to your preferred language code but english produces the best results for me
        enable_automatic_punctuation=True,
        speech_contexts=[
            speech.SpeechContext(
                phrases=[
                    "WhatsApp",
                    "send a text message to", 
                ],
                boost=20.0,
            )
        ]
    )
    return speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,
    )


def _audio_generator(
    audio_q: queue.Queue[bytes | None],
) -> Generator[bytes, None, None]:
    """Yield raw PCM bytes from the queue until the None sentinel."""
    while True:
        chunk = audio_q.get()
        if chunk is None:
            return
        yield chunk


def stream_transcribe(
    audio_q: queue.Queue[bytes | None],
    stt_client: speech.SpeechClient,
) -> None:
    """
    Runs in its own thread. Reads audio bytes from *audio_q*, streams them
    to Google STT, and prints interim / final transcripts as they arrive.
    Exits when it receives the None sentinel from the queue.
    """
    streaming_config = _make_streaming_config()

    try:
        responses = stt_client.streaming_recognize(
            config=streaming_config,
            requests=(
                speech.StreamingRecognizeRequest(audio_content=chunk)
                for chunk in _audio_generator(audio_q)
            ),
        )
        for response in responses:
            for result in response.results:
                transcript = result.alternatives[0].transcript
                tag = "[final]" if result.is_final else "[...]  "
                print(f"\r{tag} {transcript}", end="", flush=True)
                if result.is_final:
                    print()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[STT stream ended: {exc}]")


# ── Clap detection ────────────────────────────────────────────────────────────

def detect_double_clap(detector: ClapDetector, audio: bytes) -> bool:
    result = detector.run(
        thresholdBias=THRESHOLD_BIAS,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        audioData=audio,
    )
    return len(result) == 2


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.mixer.init()
    printDeviceInfo()

    stt_client = _build_stt_client()

    detector = ClapDetector(
        inputDevice=3,
        logLevel=10,
        bufferLength=CHUNK_FRAMES,
        debounceTimeFactor=0.15,
        rate=SAMPLE_RATE,
    )
    detector.initAudio()

    state = State.IDLE

    # Queue shared between the sounddevice callback and the STT thread.
    # bytes  → audio chunk to transcribe
    # None   → sentinel: shut down STT thread
    audio_q: queue.Queue[bytes | None] = queue.Queue()
    stt_thread: threading.Thread | None = None

    # Accumulate frames so we send STT_CHUNK_FRAMES at a time (smoother stream)
    pcm_buffer: bytearray = bytearray()

    def flush_to_stt(buf: bytearray, force: bool = False) -> bytearray:
        """Send complete STT_CHUNK_FRAMES-sized slices; return leftover bytes."""
        chunk_bytes = STT_CHUNK_FRAMES * 2  # int16 = 2 bytes per frame
        while len(buf) >= chunk_bytes or (force and buf):
            audio_q.put(bytes(buf[:chunk_bytes]))
            buf = buf[chunk_bytes:]
        return buf

    try:
        while True:
            # ClapDetector.getAudio() is a blocking call that yields one buffer
            raw: bytes = detector.getAudio()

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
                    args=(audio_q, stt_client),
                    daemon=True,
                )
                stt_thread.start()
                state = State.LISTENING
                continue

            # ── Forward audio to STT while listening ───────────────────────
            if state is State.LISTENING:
                pcm_buffer.extend(raw)
                pcm_buffer = flush_to_stt(pcm_buffer)

    except KeyboardInterrupt:
        print("\n[Exited gracefully]")

    finally:
        if state is State.LISTENING:
            audio_q.put(None)
            if stt_thread is not None:
                stt_thread.join()
        detector.stop()


if __name__ == "__main__":
    main()