import asyncio
import json
import random
import time
from contextlib import contextmanager
from typing import Generator

import cv2
import openai
from elevenlabs import set_api_key, generate, stream, voices
import pyaudio
import numpy
import wave

import sounddevice
import torch
from TTS.api import TTS

from transformers import pipeline, ViltForQuestionAnswering, ViltProcessor, BlipForConditionalGeneration, BlipProcessor

from llm_answers import respond_stream, make_element
from vilt_refined import yes_no_question, ask_model
from PIL import Image


class TookTooLongException(Exception):
    pass


class Snarky:
    def __init__(self, cheap: bool = True):
        assert torch.cuda.is_available()

        # Parameters
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16_000
        segment_length_silent = .5  # seconds
        self.chunk_silent = int(self.rate * segment_length_silent)
        segment_length_talking = 1  # seconds
        self.chunk_talking = int(self.rate * segment_length_talking)
        self.ambient_loudness = -1.  # max value, decreases

        self.voice = random.choice(voices())

        torch.cuda.empty_cache()

        print("Hoisting models.")
        self.vilt_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to("cuda")
        self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30, device="cuda:0")
        self.tts_model = TTS("tts_models/de/thorsten/tacotron2-DDC").to("cuda")

        openai.api_key_path = "openai_api_key.txt"
        with open("login_info.json", mode="r") as file:
            info = json.load(file)
            set_api_key(info["elevenlabs_api"])

        self.summary = ""

        self.cheap = cheap

    def reset(self) -> None:
        self.voice = random.choice(voices())
        # self.ambient_loudness = -1.
        # self.summary = ""

    @contextmanager
    def start_recording(self) -> pyaudio.Stream:
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
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
        current_frames = list[numpy.ndarray]()
        is_listening = False
        last_amplitude = None

        with self.start_recording() as stream:
            started_listening_at = time.time()

            while True:
                dynamic_chunk = self.chunk_talking if is_listening else self.chunk_silent

                data = stream.read(dynamic_chunk)
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

        stream.stop_stream()
        stream.close()
        stream._parent.terminate()

        return numpy.concatenate(current_frames, axis=0)

    def get_image(self) -> Image:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Couldn't open the webcam. What a surprise!")
            exit()

        ret, frame = cap.read()

        if not ret:
            print("Couldn't grab the photo. Again, what a surprise!")
            exit()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        cap.release()

        return pil_image

    def is_person_in_image(self, image: Image) -> bool:
        question = "Is there a person in the image?"
        return yes_no_question(image, question, self.vilt_processor, self.vilt_model)

    def is_person_close_to_camera(self, image: Image) -> bool:
        question = "Is the person close to the camera?"
        return yes_no_question(image, question, self.vilt_processor, self.vilt_model)

    def what_is_person_wearing(self, image: Image) -> str:
        question = "What is the person wearing?"
        clothes = ask_model(image, question, self.vilt_processor, self.vilt_model).lower()

        pieces = list()
        for each_piece in clothes.split("and"):
            question = f"What color is the {each_piece.strip()}?"
            each_color = ask_model(image, question, self.vilt_processor, self.vilt_model)
            colored_piece = f"{each_color.lower()} {each_piece.strip()}"
            pieces.append(colored_piece)

        return " and ".join(pieces)

    def get_image_content(self, image: Image) -> str:
        text = "a photography of"
        inputs = self.blip_processor(images=image, text=text, return_tensors="pt").to("cuda")
        out = self.blip_model.generate(**inputs, max_length=64)
        image_content = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return image_content.removeprefix(text)

    def transcribe(self, audio: numpy.ndarray) -> str:
        prediction = self.whisper_model(audio / 32_768., batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
        return prediction

    def listen(self) -> tuple[str, str]:
        sounddevice._initialize()
        audio_data = self.record_audio()
        # recorder.save_to_file(audio_data, f"output{i}.wav")

        image = self.get_image()
        image_content = self.get_image_content(image)

        text = self.transcribe(audio_data)
        return text, image_content

    def speak_tts(self, text: str) -> None:
        text = text.replace(": ", ". ")
        # text = custom_transliterate(text)

        now = time.time()
        # tts.tts_to_file(text=text, file_path="output.wav")
        wav_out = self.tts_model.tts(text=text)
        print(f"Time: {time.time() - now}")

        sample_rate = 22_000
        sounddevice.play(wav_out, sample_rate, blocking=True)

        # playsound.playsound("output.wav", True)

    def speak(self, generator: Generator[str, None, any]) -> tuple[str, str]:
        response = list()
        return_value = ""

        def _g() -> Generator[str, None, any]:
            nonlocal return_value
            while True:
                try:
                    next_item = next(generator)
                    response.append(next_item)
                    yield next_item

                except StopIteration as e:
                    return_value = e.value
                    break

        audio_stream = generate(
            text=_g(),
            voice=self.voice,
            model="eleven_multilingual_v1",
            stream=True
        )
        stream(audio_stream)

        return "".join(response), return_value

    def say(self, instruction: str, data: str | None = None, image_content: str | None = None) -> None:
        if data is None:
            data = ""

        data += make_element(image_content, "youSee")
        model = "gpt-3.5-turbo" if self.cheap else "gpt-4"
        chunks = respond_stream(instruction, data=data, recap=self.summary, model=model, temperature=.5)
        if self.cheap:
            response_chunks = list()
            while True:
                try:
                    each_chunk = next(chunks)
                    response_chunks.append(each_chunk)

                except StopIteration as e:
                    self.summary = e.value
                    break

            response = "".join(response_chunks)
            self.speak_tts(response)

        else:
            response, self.summary = self.speak(chunks)

        print(f"Snarky: {response}")


async def main() -> None:
    snarky = Snarky(cheap=True)
    # snarky = Snarky(cheap=False)

    while True:
        snarky.reset()

        image = snarky.get_image()
        if not snarky.is_person_in_image(image):
            print("No person in image.")
            continue

        print("Person in image.")
        person_description = snarky.what_is_person_wearing(image)
        image_content = snarky.get_image_content(image)

        for i in range(3):
            if snarky.is_person_close_to_camera(image):
                print("Person is close to camera.")
                snarky.say(
                    "Address the person that you see in German. "
                    "Come up with a random information that someone might ask their digital assistant for. "
                    # "Come up with a random thing a stranger in the streets might ask for. "
                    "Demand the person give you that information.",
                    image_content=image_content
                )

                for j in range(5):
                    try:
                        user_response, image_content = snarky.listen()
                        print(f"Victim: {user_response}, image content: {image_content}")
                        data = make_element(user_response, "personSays")
                        instruction = (
                            "Pick out particular aspects from what the person said. "
                            "Explicitly contest the truthfulness of these aspects. "
                            "Respond in the same language."
                        )

                        if j < 4:
                            instruction += "."
                        else:
                            instruction += " and end the conversation abruptly."
                            print("resetting...")

                    except TookTooLongException:
                        image = snarky.get_image()
                        data = None
                        if snarky.is_person_in_image(image):
                            instruction = (
                                "Use the same language as before to exclaim that you can see them. That they can stop ignoring you. "
                                "Address them directly. Ask them to come over. "
                            )

                            if j < 4:
                                instruction += ". Rephrase your initial inquiry more impolitely."
                                if j >= 1:
                                    instruction += ". Do not repeat yourself."

                            else:
                                instruction += " End the conversation angrily."
                                print("resetting...")

                        else:
                            print("Person left.")
                            print("resetting...")
                            snarky.say(
                                f"Make a snide comment about the person wearing {person_description}. "
                                f"Express that they are super impolite for simply leaving a conversation like that."
                            )
                            break

                    snarky.say(instruction, image_content=image_content, data=data)
                break

            print("Person is not close to camera.")
            instruction = f"Call over a person wearing {person_description} in German."
            if i >= 2:
                instruction += " Grow more and more impatient. Do not repeat yourself."

            snarky.say(instruction, image_content=image_content)
            time.sleep(3)
            image = snarky.get_image()
            image_content = snarky.get_image_content(image)

        else:
            print("Person is not coming over.")
            snarky.say(
                f"Make a snarky comment in German at the person wearing {person_description} for ignoring you and exclaim that you are leaving.",
                image_content=image_content
            )


if __name__ == "__main__":
    asyncio.run(main())
