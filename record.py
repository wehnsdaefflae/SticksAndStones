import time
import warnings
from contextlib import contextmanager

import playsound
import pyaudio
import numpy
import wave

import torch
import unidecode
from TTS.api import TTS

from transformers import pipeline

from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

from llm_answers import respond


class Recorder:
    def __init__(self):
        # Parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16_000
        segment_length_silent = .5  # seconds
        self.chunk_silent = int(self.rate * segment_length_silent)
        segment_length_talking = 2  # seconds
        self.chunk_talking = int(self.rate * segment_length_talking)
        self.ambient_db = 100  # max value, decreases

    @contextmanager
    def start_recording(self) -> pyaudio.Stream:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.format, channels=self.channels, rate=self.rate,
            input=True,
            # frames_per_buffer=self.chunk
        )
        try:
            yield stream

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    def get_loudness(self, data: numpy.ndarray) -> float:
        value = numpy.percentile(numpy.abs(data), 95.)  # numpy.sqrt(numpy.mean(data ** 2))
        if value == 0:  # avoid log(0)
            return 0.  # or return a very small value
        return 10. * numpy.log10(value)

    def update_threshold(self, value: float) -> None:
        self.ambient_db = self.ambient_db * .9 + value * .1

    def is_talking(self, value: float, is_already_talking: bool = False) -> bool:
        if is_already_talking:
            return value >= self.ambient_db * 1.1
        return value >= self.ambient_db * 1.3

    def save_to_file(self, all_frames: numpy.ndarray, filename: str = "output.wav") -> None:
        with wave.open(filename, mode="wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(all_frames.tobytes())
        print(f"Saved to {filename}")

    def record_audio(self) -> numpy.ndarray:
        current_frames = list[numpy.ndarray]()
        is_listening = False
        last_amplitude = None

        with self.start_recording() as stream:
            while True:
                dynamic_chunk = self.chunk_talking if is_listening else self.chunk_silent

                data = stream.read(dynamic_chunk)
                amplitude = numpy.frombuffer(data, dtype=numpy.int16)
                loudness = self.get_loudness(amplitude)

                print(f"Loudness: {loudness:.0f}, Threshold: {self.ambient_db:.0f}")

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
                        break

                last_amplitude = amplitude

        stream.stop_stream()
        stream.close()
        stream._parent.terminate()

        return numpy.concatenate(current_frames, axis=0)


def get_whisper_model() -> AutomaticSpeechRecognitionPipeline:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

    return pipe


def transcribe(audio: numpy.ndarray, pipe: AutomaticSpeechRecognitionPipeline) -> str:
    sample = {"path": "/dummy/path/file.wav", "array": audio / 32_768., "sampling_rate": 16_000}
    prediction = pipe(audio / 32_768., batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
    # prediction = pipe(sample.copy(), batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
    return prediction


def custom_transliterate(s: str) -> str:
    custom_mappings = {
        'ß': 'ss',
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue',
    }

    # Apply custom mappings
    for original, replacement in custom_mappings.items():
        s = s.replace(original, replacement)

    # Further transliterate using unidecode
    return unidecode.unidecode(s)


def speak_tts(text: str, tts: TTS) -> None:
    text = text.replace(": ", ". ")
    text = custom_transliterate(text)
    now = time.time()
    tts.tts_to_file(text=text, file_path="output.wav")

    print(f"Time: {time.time() - now}")

    playsound.playsound("output.wav", True)


def get_tts_model() -> TTS:
    # tts_models/de/thorsten/tacotron2-DCA
    # really fast!

    # tts_models/de/thorsten/vits
    # also fast. and better!

    # tts_models/de/thorsten/tacotron2-DDC
    # super fast, super good

    tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
    tts.to("cuda")
    return tts


def main() -> None:
    torch.cuda.empty_cache()

    print("Hoisting models...")
    whisper_model = get_whisper_model()
    tts_model = get_tts_model()

    recorder = Recorder()

    i = 0
    summary = ""
    while True:
        audio_data = recorder.record_audio()
        recorder.save_to_file(audio_data, f"output{i}.wav")

        text = transcribe(audio_data, whisper_model)
        print(f"in: {text}")

        # when there's a reference tro sth the user sees, respond only "look".
        # when user input contains a section "<imageContent>, respond as if you saw whatever is described in that section.
        instruction = text + (
            "Respond like a snarky, condescending, mansplaining know-it-all. Interrupt the user by starting with \"Yes yes yes...\" or other "
            "contemptuous language. Use the same language as the user.")

        # response = respond(instruction, model="gpt-3.5-turbo", data=None, recap=summary, temperature=.5)
        #response = Response(output="Yes yes yes...", summary="Yes yes yes...")

        #print(f"out: {response.output}")
        #summary = response.summary
        #speak_tts(response.output, tts_model)
        i += 1


if __name__ == "__main__":
    warnings.filterwarnings('error', category=RuntimeWarning)
    numpy.seterr(all="warn")
    assert(torch.cuda.is_available())
    main()
