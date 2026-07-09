import queue
import unittest
import wave
from types import SimpleNamespace

import numpy as np

import whisper_stt


class FakeTranscriptions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(text="  hello from Whisper  ")


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions())


class FakeTranscriber:
    def __init__(self, result: str = "hello") -> None:
        self.result = result
        self.calls = []

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.calls.append((pcm, sample_rate))
        return self.result

    def close(self) -> None:
        pass


class OpenAIWhisperTests(unittest.TestCase):
    def test_transcribe_uploads_mono_pcm_as_wav(self) -> None:
        client = FakeOpenAIClient()
        transcriber = whisper_stt.OpenAIWhisper(
            model="whisper-1",
            prompt="Jarvis",
            client=client,
        )

        text = transcriber.transcribe(np.arange(800, dtype=np.int16).tobytes(), 16000)

        self.assertEqual(text, "hello from Whisper")
        request = client.audio.transcriptions.request
        self.assertEqual(request["model"], "whisper-1")
        self.assertEqual(request["prompt"], "Jarvis")
        with wave.open(request["file"], "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnframes(), 800)

    def test_empty_audio_is_rejected_before_api_call(self) -> None:
        transcriber = whisper_stt.OpenAIWhisper(client=FakeOpenAIClient())
        with self.assertRaisesRegex(ValueError, "No audio"):
            transcriber.transcribe(b"", 16000)

    def test_unknown_backend_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "whisper_cpp.*openai"):
            whisper_stt.create_transcriber("something-else")


class StreamTranscribeTests(unittest.TestCase):
    def test_flushes_speech_when_session_ends(self) -> None:
        audio_q = queue.Queue()
        audio_q.put(np.zeros(800, dtype=np.int16).tobytes())
        audio_q.put(np.full(800, 2000, dtype=np.int16).tobytes())
        audio_q.put(None)
        transcriber = FakeTranscriber()
        transcripts = []

        whisper_stt.stream_transcribe(audio_q, transcriber, 16000, transcripts.append)

        self.assertEqual(transcripts, ["hello"])
        self.assertEqual(len(transcriber.calls), 1)
        self.assertEqual(transcriber.calls[0][1], 16000)


if __name__ == "__main__":
    unittest.main()
