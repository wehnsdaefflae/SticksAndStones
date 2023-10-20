import time
import wave
from contextlib import contextmanager

import numpy
import pyaudio
import sounddevice


class TookTooLongException(Exception):
    pass


class AudioRecorder:
    def __init__(self):
        # Recording parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16_000
        segment_length_silent = .5  # seconds
        self.chunk_silent = int(self.rate * segment_length_silent)
        segment_length_talking = 1  # seconds
        self.chunk_talking = int(self.rate * segment_length_talking)
        self.ambient_loudness = -1.  # max value, decreases

    def calibrate(self, calibration_duration: int = 3) -> None:
        """
        Calibrate the audio recorder to determine an appropriate initial value for ambient_loudness.

        Args:
            calibration_duration (int): Duration for calibration in seconds. Default is 3 seconds.
        """
        total_samples = list()
        with self.start_recording() as _stream:
            for _ in range(int(self.rate / self.chunk_silent * calibration_duration)):
                data = _stream.read(self.chunk_silent)
                amplitude = numpy.frombuffer(data, dtype=numpy.int16)
                total_samples.append(amplitude)

        calibration_data = numpy.concatenate(total_samples, axis=0)
        initial_loudness = self.get_loudness(calibration_data)
        self.ambient_loudness = initial_loudness * 1.2
        print(f"Calibration complete. Ambient loudness set to: {self.ambient_loudness:.2f}")

    @contextmanager
    def start_recording(self) -> pyaudio.Stream:
        audio = pyaudio.PyAudio()
        _stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
        )
        try:
            yield _stream
        finally:
            _stream.stop_stream()
            _stream.close()
            audio.terminate()

    def get_loudness(self, data: numpy.ndarray) -> float:
        value = numpy.percentile(numpy.abs(data), 95.)  # numpy.sqrt(numpy.mean(data ** 2))
        if value == 0:  # avoid log(0)
            return 0.  # or return a very small value
        return 10. * numpy.log10(value)

    def update_threshold(self, value: float) -> None:
        if self.ambient_loudness < 0.:
            self.ambient_loudness = value * 1.2
        else:
            self.ambient_loudness = self.ambient_loudness * .9 + value * .1

    def is_talking(self, value: float, is_already_talking: bool = False) -> bool:
        if self.ambient_loudness < 0.:
            return False
        if is_already_talking:
            return value >= self.ambient_loudness * 1.1
        return value >= self.ambient_loudness * 1.2

    def save_to_file(self, all_frames: numpy.ndarray, filename: str = "output.wav") -> None:
        with wave.open(filename, mode="wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(all_frames.tobytes())
        print(f"Saved to {filename}")

    def record_audio(self, patience_seconds: int = 10) -> numpy.ndarray:
        sounddevice._initialize()
        current_frames = list[numpy.ndarray]()
        is_listening = False
        last_amplitude = None
        with self.start_recording() as _stream:
            started_listening_at = time.time()
            while True:
                dynamic_chunk = self.chunk_talking if is_listening else self.chunk_silent
                data = _stream.read(dynamic_chunk)
                amplitude = numpy.frombuffer(data, dtype=numpy.int16)
                loudness = self.get_loudness(amplitude)
                print(f"Loudness: {loudness:.0f}, Threshold: {self.ambient_loudness:.0f}")
                if self.is_talking(loudness, is_already_talking=is_listening):
                    if not is_listening:
                        print("STARTED LISTENING")
                        if last_amplitude is not None:
                            current_frames.append(last_amplitude)
                        is_listening = True
                    current_frames.append(amplitude)
                else:
                    self.update_threshold(loudness)
                    if is_listening and 0 < len(current_frames):
                        print("STOPPED LISTENING")
                        sounddevice.stop()
                        break
                    if patience_seconds < time.time() - started_listening_at:
                        print("Person not talking.")
                        raise TookTooLongException()
                last_amplitude = amplitude
        _stream.stop_stream()
        _stream.close()
        _stream._parent.terminate()
        audio_data = numpy.concatenate(current_frames, axis=0)
        return audio_data
