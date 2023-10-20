import asyncio
import datetime
import json
import pathlib
import random
import time
from contextlib import contextmanager
from typing import Generator
from loguru import logger

import cv2
import openai
from elevenlabs import set_api_key, generate, stream, voices
import pyaudio
import numpy
import wave

import sounddevice
import torch

from transformers import pipeline, ViltForQuestionAnswering, ViltProcessor, BlipForConditionalGeneration, BlipProcessor

from llm_answers import make_element
from vilt_refined import yes_no_question, ask_model
from PIL import Image


class TookTooLongException(Exception):
    pass


class Snarky:
    def __init__(self):
        assert torch.cuda.is_available()

        now = datetime.datetime.now()
        self.messages_log = pathlib.Path(f"messages_{now.strftime('%Y%m%d_%H%M%S')}.jsonl")

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

        openai.api_key_path = "openai_api_key.txt"
        with open("login_info.json", mode="r") as file:
            info = json.load(file)
            set_api_key(info["elevenlabs_api"])

        self.summary = ""

        self.messages = list()

    def _append_message_log(self, message: dict[str, str]) -> None:
        with self.messages_log.open(mode="a") as file:
            file.write(json.dumps(message) + "\n")

    def reset(self) -> None:
        self.voice = random.choice(voices())
        # self.ambient_loudness = -1.
        # self.summary = ""
        self.messages.clear()
        instructions = (
            "You're a user asking a question to a digital assistant like Alexa, Siri, or Google Assistant. You're very sceptical of all the answers you get. You come "
            "up with reasons why you don't believe the information you get. You're quite impolite.\n"
            "You start each conversation in German. After the person responded, you switch to their language.\n"
            "You see an image of the person you're talking to. The content of the image is described to you in the `ImageContent` XML tag. From time to time, you "
            "rudely comment on changes from one image to another.\n"
            "You never ask what you can do for the person. Instead you expect them to be do things for you.\n"
            "You never respond by saying that you're an AI or an assistant. If you cannot respond, you reply as if you were a human finding reasons to avoid the topic."
        )

        system_prompt = {
            "role": "system",
            "content": instructions
        }
        self._append_message_log(system_prompt)
        self.messages.append(system_prompt)

    @contextmanager
    def start_recording(self) -> pyaudio.Stream:
        audio = pyaudio.PyAudio()
        _stream = audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            # frames_per_buffer=self.chunk
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
        # snarky.save_to_file(audio_data, f"output{i}.wav")

        return audio_data

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
        text = "the image definitely shows"
        # text = "a photography of"
        inputs = self.blip_processor(images=image, text=text, return_tensors="pt").to("cuda")
        out = self.blip_model.generate(**inputs, max_length=64)
        image_content = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return image_content.removeprefix(text)

    def transcribe(self, audio: numpy.ndarray) -> str:
        prediction = self.whisper_model(audio / 32_768., batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
        return prediction

    def speak(self, generator: Generator[str, None, any]) -> str:
        return_value = ""

        def _g() -> Generator[str, None, any]:
            nonlocal return_value
            while True:
                try:
                    yield next(generator)

                except StopIteration as e:
                    return_value = e.value
                    break

        audio_stream = generate(
            text=_g(),
            voice=self.voice,
            model="eleven_multilingual_v1",
            stream=True,
            latency=3
        )
        stream(audio_stream)

        return return_value

    def _respond(self, message: str, *args: any, **kwargs) -> Generator[str, None, str]:
        input_message = {"role": "user", "content": message}
        self._append_message_log(input_message)
        self.messages.append(input_message)

        full_output = list()
        for chunk in openai.ChatCompletion.create(*args, messages=self.messages, stream=True, **kwargs):
            content = chunk["choices"][0].get("delta", dict()).get("content")
            if content is not None:
                full_output.append(content)
                print(content, end="", flush=True)
                yield content  # Yielding the content for each chunk

        output = "".join(full_output)

        output_message = {"role": "assistant", "content": output}
        self._append_message_log(output_message)
        self.messages.append(output_message)

        while len(self.messages) >= 11:
            self.messages.pop(1)
        return output

    def say(self, instruction: str, image_content: str) -> str:
        full_prompt = make_element(image_content, "ImageContent") + instruction
        model = "gpt-4"
        chunks = self._respond(full_prompt, model=model, temperature=.0)
        response = self.speak(chunks)
        return response


async def call_over(image_content: str, person_description: str, snarky: Snarky, attempt: int) -> None:
    full_prompt = f"Call over the person in the image. You have a question for them. Address them by their clothing: {person_description}."
    if attempt >= 1:
        full_prompt += f" This is attempt number {attempt + 1}. You're growing impatient."

    snarky.say(
        f"[{full_prompt}]",
        image_content=image_content
    )


async def outraged(image_content: str, person_description: str, snarky: Snarky) -> None:
    snarky.say(
        f"Make a snide remark about the person wearing {person_description}. "
        f"Express that they are super impolite for simply leaving a conversation like that. "
        f"End the conversation angrily.",
        image_content=image_content
    )


async def point_out(image_content: str, person_description: str, snarky: Snarky) -> None:
    snarky.say(
        f"Use the same language as before to exclaim that you can see them. "
        f"That they can stop ignoring you. "
        f"Address them directly by their clothes: {person_description}. Ask them to come over. "
        f"Rephrase your initial request more impolitely.",
        image_content=image_content,
    )


async def point_out_again(image_content: str, person_description: str, snarky: Snarky) -> None:
    snarky.say(
        f"Use the same language as before to exclaim that you can see them. "
        f"That they can stop ignoring you. "
        f"Address them directly by their clothes: {person_description}. Ask them to come over. "
        f"Rephrase your initial request more impolitely. "
        f"Don't repeat anything you've already said.",
        image_content=image_content
    )


async def point_out_goodbye(image_content: str, person_description: str, snarky: Snarky) -> None:
    snarky.say(
        f"Use the same language as before to exclaim that you can see them. "
        f"Address them directly by their clothes: {person_description}. "
        f"Say that it is very impolite to ignore you and end the conversation angrily.",
        image_content=image_content
    )


async def finish(image_content: str, snarky: Snarky) -> None:
    snarky.say("Pick out particular aspects of what the person said. "
               "Explicitly doubt the truthfulness of these aspects. "
               "Don't repeat anything you've already said. "
               "End the conversation abruptly. Respond in the same language.",
               image_content=image_content)


async def pick(image_content: str, snarky: Snarky) -> None:
    snarky.say(
        "Pick out particular aspects of what the person said. "
        "Do not literally repeat what they said. "
        "Don't repeat anything you've already said. "
        "Explicitly doubt the truthfulness of these aspects. "
        "Respond in the same language.",
        image_content=image_content)


async def initiate(person_description: str, image_content: str, snarky: Snarky) -> None:
    snarky.say(
        f"Address the person wearing {person_description} in German. "
        f"Come up with a random information that someone might ask their digital assistant for. "
        f"Do not ask for the weather. "
        # f"Come up with a random thing a stranger in the streets might ask for. "
        f"Demand the person give you that information.",
        image_content=image_content
    )


async def main() -> None:
    # todo:
    # - [ ] change it to
    #   - [ ] system prompt
    #           "you're the user of a digital assistant. asking questions to the assistant.
    #           you're very sceptical of all the answers you get. you're also quite impolite.
    #           you see the person in the image. you comment on changes from one image to the next."
    #   - [ ] classical history

    snarky = Snarky()
    # calibrate audio level?

    while True:
        snarky.reset()

        image = snarky.get_image()
        while not snarky.is_person_in_image(image):
            logger.info("No person in image.")
            image = snarky.get_image()

        person_description = snarky.what_is_person_wearing(image)
        logger.info("Person in image.")

        person_left = False
        attempt = 0
        while not snarky.is_person_close_to_camera(image):
            logger.info("Person not close.")

            if not snarky.is_person_in_image(image):
                person_left = True
                break

            if attempt >= 3:
                break

            person_description = snarky.what_is_person_wearing(image)
            image_content = snarky.get_image_content(image)
            logger.info(f"Attempt {attempt + 1} at calling over person wearing {person_description} in image {image_content}.")
            await call_over(image_content=image_content, person_description=person_description, snarky=snarky, attempt=attempt)

            attempt += 1
            image = snarky.get_image()

        if attempt >= 3:
            logger.info("Person ignores.")
            image_content = snarky.get_image_content(image)
            person_description = snarky.what_is_person_wearing(image)
            snarky.say(f"[Complain that the person wearing {person_description} ignores you.]", image_content=image_content)
            continue

        if person_left:
            logger.info("Person left.")
            image_content = snarky.get_image_content(image)
            snarky.say(f"[Complain that the person wearing {person_description} does not interact.]", image_content=image_content)
            continue

        logger.info("Person is close")

        try:
            user_response = "[Ask the person like a user might ask something from their digital assistant.]"
            while True:
                image = snarky.get_image()
                image_content = snarky.get_image_content(image)

                logger.info(f"Snarky responds to \"{user_response}\" from person in image {image_content}.")
                snarky_says = snarky.say(user_response, image_content=image_content)

                logger.info("listening...")
                audio_data = snarky.record_audio()
                user_response = snarky.transcribe(audio_data)

        except TookTooLongException:
            image_content = snarky.get_image_content(image)
            person_description = snarky.what_is_person_wearing(image)
            logger.info(f"Person wearing {person_description} in image {image_content} does not respond.")

            snarky.say(f"[Complain that the person wearing {person_description} seems a bit slow.]", image_content=image_content)
            continue


if __name__ == "__main__":
    asyncio.run(main())
