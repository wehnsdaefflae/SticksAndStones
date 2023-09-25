import numpy
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from datasets import Audio, load_dataset
import torch


def translate() -> None:
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="translate")

    # load streaming dataset and read first audio sample
    ds = load_dataset("common_voice", "fr", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    input_speech = next(iter(ds))["audio"]
    input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)


def chunk() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]

    prediction = pipe(sample.copy(), batch_size=8)["text"]

    # we can also return timestamps for the predictions
    prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]


def transcribe(audio: numpy.ndarray) -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        device=device,
    )

    sample = {"path": "/dummy/path/file.wav", "array": audio / 32_768., "sampling_rate": 16_000}
    prediction = pipe(sample.copy(), batch_size=8, generate_kwargs={"task": "transcribe", "language": "german"})["text"]
    return prediction


if __name__ == "__main__":
    chunk()
