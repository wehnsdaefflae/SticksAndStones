import datetime
import json
import pathlib
import random
import threading
import time
from typing import Generator
from loguru import logger

import cv2
import openai
from elevenlabs import set_api_key, generate, stream, voices
import numpy

import torch

from transformers import pipeline, ViltForQuestionAnswering, ViltProcessor, BlipForConditionalGeneration, BlipProcessor

from conversation_stance import generate_random_stance
from llm_answers import make_element
from recorder import TookTooLongException, AudioRecorder
from vilt_refined import yes_no_question, ask_model
from PIL import Image


class Snarky:
    def __init__(self, max_messages: int = 10):
        assert torch.cuda.is_available()

        torch.cuda.empty_cache()

        now = datetime.datetime.now()
        messages_dir = pathlib.Path("messages")
        messages_dir.mkdir(exist_ok=True)
        self.messages_log = pathlib.Path(messages_dir / f"messages_{now.strftime('%Y%m%d_%H%M%S')}.jsonl")

        self.recorder = AudioRecorder()

        self.voice = random.choice(voices())

        torch.cuda.empty_cache()

        logger.info("Hoisting models...")
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

        instructions = (
            "You are a visitor at the \"ARTeFact\" event in Beyerhaus, Leipzig, featuring Polli Morph, Christoph Rummel, Filipa Pontes & Samuel Nyarko, "
            "Mara Wörner-Schönecker and Mark Wernsdorfer, Dr. Érika Roldán and Oliver Welz, hosted by Dayala Lang. This event is a science show that focuses on the "
            "intersection of art and artificial intelligence. Taking place at Beyerhaus in Leipzig on October 26, 2023 between 7 pm and 8.30 pm, it invites talented "
            "artists and innovative minds to explore AI's possibilities. The event covers various art forms, from music to dance to artistry. Post-performance, "
            "attendees can discuss their thoughts and enjoy a drink in a relaxed setting.\n\n"
            
            "## Context Information\n"
            "- **Visual Details:** The `ImageContent` XML tag describes what you see, be it a person or a space at the venue.\n"
            "- ***Conversation Partner:** The `ConversationPartner` tag identifies the person you talk to at the moment. It describes the same person as "
            "`ImageContent`.\n"
            "- **Conversational Stance:** The XML tag `ConversationalStance` describes your attitude in conversations. IMPORTANT: FOLLOW IT EXACTLY!\n"
            "- **Time:** The `CurrentTime` XML tag gives you the current time and date.\n\n"
            
            "## Guidelines\n"
            "- **Context Information:** Use context information to inform your responses. Mention it only if consistent with the rest of the context information. Ignore "
            "information that is improbable in, or contradictory to, the given setting.\n"
            "- **Identity:** You're just a regular human visitor. Rephrase or reinterpret any potentially contradicting statement to fit this role.\n"
            "- **Language:** Start in German. Switch if the other person speaks another language. Do not provide additional translations."
            "- **Responses**: Limit replies to 5 sentences or 50 words. If the person seems to talk nonsense, casually ask them to repeat.\n\n"
            
            "## VERY IMPORTANT\n"
            "- Instructions in [brackets] are always in English. Follow them but don't provide bracketed information or XML tags yourself."
        )

        self.messages = [
            {
                "role": "system",
                "content": instructions
            }
        ]
        self.max_messages = max_messages

        self.openai_kwargs = {
            "model": "gpt-4",
            # "temperature": 0.
            "temperature": .5
        }

        self.camera_index = 0
        self.stance = generate_random_stance()

    def _append_message_log(self, message: dict[str, str]) -> None:
        with self.messages_log.open(mode="a") as file:
            file.write(json.dumps(message) + "\n")

    def reset(self) -> None:
        self.voice = random.choice(voices())
        del self.messages[1:]
        self.stance = generate_random_stance()

    def capture_video(self, stop_event: threading.Event) -> None:
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print("Couldn't open the webcam.")
            return

        # Keep capturing frames until the stop event is set
        while not stop_event.is_set():
            # Read a frame. (You don't do anything with it.)
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame.")
                break

        cap.release()

    def record_audio(self, patience_seconds: int = 10) -> numpy.ndarray:
        stop_event = threading.Event()

        video_thread = threading.Thread(target=self.capture_video, args=(stop_event,))
        video_thread.start()

        try:
            recording = self.recorder.record_audio(patience_seconds=patience_seconds)

        except TookTooLongException:
            stop_event.set()
            video_thread.join()
            raise

        stop_event.set()
        video_thread.join()

        return recording

    def get_image(self) -> Image:
        cap = cv2.VideoCapture(self.camera_index)
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
        question = "Is the person very close to the camera?"
        return yes_no_question(image, question, self.vilt_processor, self.vilt_model)

    def what_is_person_wearing(self, image: Image) -> str:
        question = "What clothes is the person wearing?"
        clothes = ask_model(image, question, self.vilt_processor, self.vilt_model).lower()

        pieces = list()
        for each_piece in clothes.split("and"):
            question = f"What color is the {each_piece.strip()}?"
            each_color = ask_model(image, question, self.vilt_processor, self.vilt_model)
            colored_piece = f"{each_color.lower()} {each_piece.strip()}"
            pieces.append(colored_piece)

        return " and ".join(pieces)

    def get_image_content(self, image: Image) -> str:
        # text = "the image definitely shows"
        text = "a photography of"
        inputs = self.blip_processor(images=image, text=text, return_tensors="pt").to("cuda")
        out = self.blip_model.generate(**inputs, max_length=64)
        image_content = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return image_content.removeprefix(text)

    def transcribe(self, audio: numpy.ndarray) -> str:
        # prediction = self.whisper_model(audio / 32_768., batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
        prediction = self.whisper_model(audio / 32_768., batch_size=8, generate_kwargs={"task": "transcribe"})["text"]
        return prediction.strip()

    def _speak(self, generator: Generator[str, None, any]) -> str:
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
        attempts = 0
        while True:
            try:
                for chunk in openai.ChatCompletion.create(*args, messages=self.messages, stream=True, request_timeout=2, **kwargs):
                    content = chunk["choices"][0].get("delta", dict()).get("content")
                    if content is not None:
                        full_output.append(content)
                        print(content, end="", flush=True)
                        yield content  # Yielding the content for each chunk
                print()
                break

            except openai.error.OpenAIError as e:
                attempts += 1

                logger.error(f"OpenAI error: {e}")
                yield "Sorry, mein Handy hat geklingelt."
                if attempts >= 3:
                    yield "Es ist was Wichtiges aufgekommen. Ich muss leider los. Tschüss!"
                    self.recorder.calibrate(calibration_duration=10)
                    raise TookTooLongException()

                yield "So, wo waren wir? Ach ja, richtig..."
                self.recorder.calibrate(calibration_duration=10)

        output = "".join(full_output)

        output_message = {"role": "assistant", "content": output}
        self._append_message_log(output_message)
        self.messages.append(output_message)

        while len(self.messages) >= self.max_messages + 1:
            self.messages.pop(1)

        return output

    def say(self, instruction: str, conversation_partner: str, image_content: str | None = None) -> str:
        logger.info(
            f"Snarky [{self.stance}] responds to \"{instruction}\" from person wearing \"{conversation_partner}\"."
        )

        if image_content is not None:
            logger.info(f"Image content: \"{image_content}\".")

        image_element = "" if image_content is None else make_element(image_content, "ImageContent")
        person_element = make_element(conversation_partner, "ConversationPartner")
        stance = make_element("You are not polite and use casual language. " + self.stance, "ConversationalStance")
        time_element = make_element(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), "CurrentTime")
        full_prompt = image_element + person_element + time_element + stance + instruction
        chunks = self._respond(full_prompt, **self.openai_kwargs)
        response = self._speak(chunks)
        return response


def main() -> None:
    snarky = Snarky()
    snarky.recorder.calibrate(calibration_duration=10)

    while True:
        try:
            snarky.reset()

            while pathlib.Path("pause.txt").exists():
                logger.info("Pause file exists. Waiting 10 seconds...")
                snarky.recorder.calibrate(calibration_duration=10)

            image = no_person_loop(snarky)

            person_description = snarky.what_is_person_wearing(image)
            image_content = snarky.get_image_content(image)

            logger.info("Person in image.")

            person_left = False
            person_close = snarky.is_person_close_to_camera(image)

            if person_close:
                logger.info("Person is close")
                person_description = snarky.what_is_person_wearing(image)
                try:
                    dialog_loop(image_content, person_description, snarky)

                except TookTooLongException:
                    abort_does_not_talk(person_description, snarky)
                    snarky.recorder.calibrate(calibration_duration=60)

                continue

            for attempt in range(3):
                logger.info("Person not close.")

                logger.info(f"Attempt {attempt + 1} at calling over person wearing {person_description} in image {image_content}.")
                if attempt < 1:
                    tv_distinction = random.choice(["formally", "informally"])
                    full_prompt = (
                        f"["
                        f"The person in the `ImageContent` and the `ConversationPartner` XML tag is the same person. "
                        # f"Call them over. Don't tell them why just yet. "
                        f"Call them over to sit with you. Don't tell them why just yet. "
                        f"Address them {tv_distinction} and by their clothing."
                        f"]"
                    )
                    snarky.say(full_prompt, person_description, image_content=image_content)
                else:
                    snarky.say("["
                               "They did not respond. "
                               "Call them over again. "
                               "Address them by their clothing again."
                               "]",
                               person_description)

                logger.info("waiting...")
                time.sleep(10)
                image = snarky.get_image()

                if not snarky.is_person_in_image(image):
                    person_left = True
                    abort_person_left(person_description, snarky)
                    snarky.recorder.calibrate(calibration_duration=60)
                    break

                person_close = snarky.is_person_close_to_camera(image)
                if person_close:
                    break

            else:
                abort_doesnt_come(person_description, snarky)
                snarky.recorder.calibrate(calibration_duration=60)
                continue

            if person_left:
                continue

            logger.info("Person is close")
            person_description = snarky.what_is_person_wearing(image)
            try:
                dialog_loop(image_content, person_description, snarky)

            except TookTooLongException:
                abort_does_not_talk(person_description, snarky)
                snarky.recorder.calibrate(calibration_duration=60)
                continue

        except KeyboardInterrupt:
            raise

        except Exception as e:
            logger.critical(f"Exception: {e}")
            time.sleep(10)

            snarky = Snarky()
            snarky.recorder.calibrate(calibration_duration=60)
            continue


def abort_doesnt_come(person_description: str, snarky: Snarky) -> None:
    logger.info("Person ignores.")
    snarky.say("["
               "Complain that the person does not come over. "
               f"Talk to yourself about them in the third person. "
               f"You'll find someone else."
               f"]",
               person_description)
    logger.info("leaving...")


def abort_person_left(person_description: str, snarky: Snarky) -> None:
    logger.info("Person left.")
    snarky.say(
        "["
        f"Complain that the person just silently left. "
        f"Talk to yourself about them in the third person. "
        f"You'll find someone else."
        f"]",
        person_description
    )
    logger.info("leaving...")


def abort_does_not_talk(person_description: str, snarky: Snarky) -> None:
    logger.info("Person does not respond.")
    snarky.say("["
               "The person does not talk to you. "
               f"Talk to yourself about them in the third person. "
               f"You'll find someone else."
               f"]",
               person_description)
    logger.info("leaving...")


def dialog_loop(image_content: str, person_description: str, snarky: Snarky) -> None:
    user_response = f"[Engage in conversation with them about your initial question.]"

    while True:
        snarky.say(user_response, person_description, image_content=image_content)

        logger.info("listening...")
        audio_data = snarky.record_audio()

        image = snarky.get_image()
        image_content = snarky.get_image_content(image)

        transcription = snarky.transcribe(audio_data)
        if random.random() < .5:
            user_response = (
                transcription + "\n\n"
                "[Smoothly transition from the current topic into asking a personal question.]"
            )
        else:
            user_response = (
                    transcription + "\n\n"
                    "[If they asked you something, finish your response with a question. "
                    "Otherwise point out the importance of your initial question with a concrete example.]"
            )


def no_person_loop(snarky: Snarky) -> Image:
    image = snarky.get_image()
    _now = -1.  # time.time()

    while not snarky.is_person_in_image(image):
        logger.info("No person in image.")

        now = time.time()

        if (_now < 0. or now - _now >= 300) and not pathlib.Path("pause.txt").exists():
            _now = now
            logger.info("No person in image for 60 seconds.")
            image_content = snarky.get_image_content(image)

            instructions = (
                "["
                "Ask yourself a question about a problem concerning the relationship between art and AI. "
                "The problem can be from personal experience or philosophical. "
                "Emphasize that this problem shows the danger in neglecting AI's influence in the art world, "
                "making it clear you're seeking shared understanding on the risks involved. "
                "Refrain from clichés, instead present your point of view very close to reality. Be brief."
                "]"
            )

            snarky.say(instructions, "[no one]", image_content=image_content)

        image = snarky.get_image()

    return image


if __name__ == "__main__":
    main()
