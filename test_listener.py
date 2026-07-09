import io
import unittest
from contextlib import redirect_stdout

import listener


class FakePyAudio:
    def __init__(self, default_index=1) -> None:
        self.default_index = default_index
        self.terminated = False
        self.devices = [
            {
                "name": "Yeyito Microphone",
                "maxInputChannels": 1,
                "defaultSampleRate": 48000,
            },
            {
                "name": "MacBook Pro Microphone",
                "maxInputChannels": 1,
                "defaultSampleRate": 48000,
            },
            {
                "name": "MacBook Pro Speakers",
                "maxInputChannels": 0,
                "defaultSampleRate": 48000,
            },
        ]

    def get_device_count(self):
        return len(self.devices)

    def get_device_info_by_index(self, index):
        return self.devices[index]

    def get_default_input_device_info(self):
        if self.default_index is None:
            raise OSError("no default")
        return {"index": self.default_index}

    def terminate(self):
        self.terminated = True


class InputDeviceTests(unittest.TestCase):
    def resolve(self, selection, audio):
        with redirect_stdout(io.StringIO()):
            return listener.resolve_input_device(selection, lambda: audio)

    def test_uses_default_input_device(self) -> None:
        audio = FakePyAudio(default_index=1)
        self.assertEqual(self.resolve("", audio), 1)
        self.assertTrue(audio.terminated)

    def test_resolves_name_case_insensitively(self) -> None:
        audio = FakePyAudio()
        self.assertEqual(self.resolve("yeyito microphone", audio), 0)

    def test_rejects_output_only_device(self) -> None:
        audio = FakePyAudio()
        with self.assertRaisesRegex(ValueError, "not an available input device"):
            self.resolve("2", audio)
        self.assertTrue(audio.terminated)

    def test_falls_back_to_first_input_without_system_default(self) -> None:
        audio = FakePyAudio(default_index=None)
        self.assertEqual(self.resolve("", audio), 0)


if __name__ == "__main__":
    unittest.main()
