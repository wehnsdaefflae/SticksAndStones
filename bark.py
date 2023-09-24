import os
import time

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoProcessor, AutoModel
from transformers.models.bark.modeling_bark import BarkModel
import scipy
import playsound
import optimum
import accelerate
# import pygobject


def speak(text: str, model: BarkModel) -> None:
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
    os.environ["SUNO_OFFLOAD_CPU"] = "True"
    os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float)
    # model = BarkModel.from_pretrained("suno/bark-small", torch_dtype=torch.float)
    model.to("cuda")
    model = BetterTransformer.transform(model, keep_original_model=False)

    # model.enable_cpu_offload()

    t = "Hallo, mein Name ist Suno. Und, äh - und ich mag Pizza. [lacht] Aber ich habe auch andere Interessen, zum Beispiel Tic Tac Toe spielen."
    speak(t, model)


if __name__ == "__main__":
    # https://github.com/haoheliu/versatile_audio_super_resolution/blob/main/audiosr/__main__.py
    # https://huggingface.co/blog/optimizing-bark#optimization-techniques
    main()
