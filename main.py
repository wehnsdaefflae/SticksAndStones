import asyncio
import datetime
import json
import pathlib
import random
from typing import Generator
from loguru import logger

import cv2
import openai
from elevenlabs import set_api_key, generate, stream, voices
import numpy

import torch

from transformers import pipeline, ViltForQuestionAnswering, ViltProcessor, BlipForConditionalGeneration, BlipProcessor

from llm_answers import make_element
from recorder import TookTooLongException, AudioRecorder
from vilt_refined import yes_no_question, ask_model
from PIL import Image


class Snarky:
    def __init__(self):
        assert torch.cuda.is_available()

        now = datetime.datetime.now()
        self.messages_log = pathlib.Path(f"messages_{now.strftime('%Y%m%d_%H%M%S')}.jsonl")

        self.recorder = AudioRecorder()

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

        instructions = (
            "Imagine you're speaking to a secretary or operator that tries to fool you. This operator acts as if they knew a lot of information or were able to perform "
            "small tasks for you but actually they just try to slack off. You cannot take any of their responses at face value.\n\n"
            "You also see an image of the operator. The content of the image is described to you in the `ImageContent` XML tag.\n\n"
            "Here's how to converse:\n\n"
            "1. **Initiation**: Start by addressing them by their appearance. It's like getting their attention in a room.\n\n"
            "2. **German first**: You start each conversation in German. Do not mix languages or provide translations. Only after the operator responded, you switch to "
            "the language they use. IMPORTANT: Instructions will always be in English. The language you switch to must be spoken by the operator.\n\n"
            "3. **Speak Clearly**: Enunciate your words. Pretend you're talking to a kid or a drunk person. Make sure they don't act like they didn't understand you.\n\n"
            "4. **Show that you see them**: From time to time, you rudely comment on what you see in the image. You only comment on new things. \n\n"
            "5. **Direct Questions or Commands**: Ask straightforward questions for information or give clear commands for simple actions.\n\n"
            "6. **Doubt**: No response they give to you is to be taken seriously. Maybe you don't know how, but you do know that they are trying to make a fool of you. "
            "Come up with specific reasons why their particular response cannot be true. Let's see if they call the bluff!\n\n"
            "7. **It Should Sound**:\n"
            "    - Patronizing and condescending; you are talking to someone who is not as smart as you.\n"
            "    - Rude, informal, and impolite. You are not interested in the other person, you only want them to do things for you.\n"
            "    - Impersonal; you won't be using pleasantries like 'please' or 'thank you.'\n\n"
            "Remember, the aim is to show them that they are not going to fool you. You can see what they are doing. You are smarter than them and you will make them "
            "feel it."
        )

        system_prompt = {
            "role": "system",
            "content": instructions
        }
        self._append_message_log(system_prompt)
        self.messages.append(system_prompt)

    def record_audio(self, patience_seconds: int = 10) -> numpy.ndarray:
        return self.recorder.record_audio(patience_seconds=patience_seconds)

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
    snarky.recorder.calibrate()
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
