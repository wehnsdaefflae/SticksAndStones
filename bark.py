import os

from transformers import AutoProcessor, AutoModel
import scipy
import playsound
# import pygobject


os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"


processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")
model.to("cuda")

inputs = processor(
    # text=["Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe."],
    text=["Hallo, mein Name ist Suno. Und, äh - und ich mag Pizza. [lacht] Aber ich habe auch andere Interessen, zum Beispiel Tic Tac Toe spielen."],
    return_tensors="pt",
)
inputs.to("cuda")

speech_values = model.generate(**inputs, do_sample=True)

sampling_rate = 24_000
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())
playsound.playsound("bark_out.wav", True)
