import time

import playsound
import pyaudio
import numpy
import wave

import torch
import unidecode
from TTS.api import TTS

from llm_answers import respond
from transformers import pipeline

from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline


# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16_000
CHUNK = int(RATE * .5)  # seconds
SILENCE_THRESHOLD = 1_000  # Experiment with this value


def start_recording() -> pyaudio.Stream:
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE,
        input=True, frames_per_buffer=CHUNK
    )
    return stream


def is_silent(data: numpy.ndarray) -> bool:
    value = numpy.percentile(numpy.abs(data), 95.)
    print(value)
    return value < SILENCE_THRESHOLD


def save_to_file(all_frames: numpy.ndarray, filename: str = "output.wav") -> None:
    with wave.open(filename, mode="wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(all_frames.tobytes())
    print(f"Saved to {filename}")


def record_audio() -> numpy.ndarray:
    current_frames = list[numpy.ndarray]()

    is_listening = False
    stream = start_recording()
    while True:
        data = stream.read(CHUNK)
        amplitude = numpy.frombuffer(data, dtype=numpy.int16)

        if is_silent(amplitude):
            if is_listening:
                current_frames.append(amplitude)
                break

        else:
            if not is_listening:
                is_listening = True

            current_frames.append(amplitude)

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

    i = 0
    summary = ""
    while True:
        audio_data = record_audio()
        save_to_file(audio_data, f"output{i}.wav")
        text = transcribe(audio_data, whisper_model)
        print(f"in: {text}")
        instruction = text + " Respond in the user's language and in a snarky and condescending way."
        response = respond(instruction, model="gpt-3.5-turbo", data=None, recap=summary, temperature=.5)
        print(f"out: {response.output}")
        summary = response.summary
        speak_tts(response.output, tts_model)
        i += 1


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
