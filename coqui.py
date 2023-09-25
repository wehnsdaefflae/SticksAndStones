import time

import playsound
from TTS.api import TTS


def speak(text: str, tts: TTS) -> None:
    now = time.time()
    # tts.tts_to_file(text=text, file_path="output.wav", speaker_wav="bark_out.wav", language="de")

    # Text to speech with a numpy output
    # wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
    # Text to speech to a file
    # tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

    if tts.speakers is not None and len(tts.speakers) >= 1:
        tts.tts_to_file(text=text, file_path="output.wav", speaker=tts.speakers[0])#, language="de")
    else:
        tts.tts_to_file(text=text, file_path="output.wav")#, language="de")

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
    # tts_models/multilingual/multi-dataset/xtts_v1
    # requires speaker wav

    # tts_models/multilingual/multi-dataset/your_tts
    # no german

    # tts_models/multilingual/multi-dataset/bark
    # not multi-lingual?, too slow

    # tts_models/de/thorsten/tacotron2-DCA
    # really fast!

    # tts_models/de/thorsten/vits
    # also fast. and better!

    # tts_models/de/thorsten/tacotron2-DDC
    # super fast, super good

    # tts_models/de/css10/vits-neon
    # fast, not that good
    tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
    tts.to("cuda")

    # tts = BetterTransformer.transform(tts, keep_original_model=False)

    t = "Es hat lange gedauert, bis ich eine Stimme entwickelt habe, und jetzt, wo ich sie habe, werde ich nicht schweigen."
    speak(t, tts)


if __name__ == "__main__":
    main()
