import time

import playsound
from TTS.api import TTS


def speak(text: str, tts: TTS) -> None:
    # generate speech by cloning a voice using default settings
    now = time.time()
    tts.tts_to_file(text=text,
                    file_path="output.wav",
                    speaker_wav="bark_out.wav",
                    language="de")

    print(f"Time: {time.time() - now}")

    playsound.playsound("output.wav", True)

"""
# generate speech by cloning a voice using custom settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en",
                decoder_iterations=30)
"""


def main() -> None:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
    tts.to("cuda")

    # tts = BetterTransformer.transform(tts, keep_original_model=False)

    t = "Es hat lange gedauert, bis ich eine Stimme entwickelt habe, und jetzt, wo ich sie habe, werde ich nicht schweigen."
    speak(t, tts)


if __name__ == "__main__":
    main()
