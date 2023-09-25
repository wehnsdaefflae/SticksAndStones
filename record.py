import os
import time

import playsound
import pyaudio
import numpy
import wave

import scipy
import torch

from llm_answers import respond
from transformers import pipeline, AutoProcessor

from optimum.bettertransformer import BetterTransformer
from transformers import AutoModel

from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
from transformers.models.bark.modeling_bark import BarkModel


# Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
# RATE = 44_100
RATE = 16_000
CHUNK = int(RATE * 1.)  # seconds
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


def get_bark_model() -> BarkModel:
    os.environ["SUNO_OFFLOAD_CPU"] = "True"
    os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float)
    # model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float)
    model.to("cuda")
    model = BetterTransformer.transform(model, keep_original_model=False)
    return model


def transcribe(audio: numpy.ndarray, pipe: AutomaticSpeechRecognitionPipeline) -> str:
    sample = {"path": "/dummy/path/file.wav", "array": audio / 32_768., "sampling_rate": 16_000}
    prediction = pipe(audio / 32_768., batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
    # prediction = pipe(sample.copy(), batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
    return prediction


def speak(text: str, model: BarkModel) -> None:
    # https://app.coqui.ai/account
    # or coqui locally
    now = time.time()

    processor = AutoProcessor.from_pretrained("suno/bark-small")

    inputs = processor(
        # text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
        text=[text],
        return_tensors="pt",
        # voice_preset="v2/de_speaker_4",
        # voice_preset="v2/de_speaker_6",
        voice_preset="v2/de_speaker_7",
        # voice_preset="v2/de_speaker_8",
    )
    inputs.to("cuda")

    speech_values = model.generate(**inputs, do_sample=True)

    sampling_rate = 24_000
    scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
    print(f"Time: {time.time() - now}")

    playsound.playsound("bark_out.wav", True)


def main() -> None:
    print("Hoisting models...")
    whisper_model = get_whisper_model()
    bark_model = get_bark_model()

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
        speak(response.output, bark_model)
        i += 1


if __name__ == "__main__":
    main()
