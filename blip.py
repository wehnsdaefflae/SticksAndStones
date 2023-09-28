import requests
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def get_image() -> Image:
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


def get_blip_model() -> BlipForConditionalGeneration:
    return BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


def get_blip_processor() -> BlipProcessor:
    return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")


def main() -> None:
    raw_image = get_image()

    # raw_image.show()

    # conditional image captioning
    text = "a photography of"

    processor = get_blip_processor()
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    model = get_blip_model()
    out = model.generate(**inputs, max_length=64)

    image_content = processor.decode(out[0], skip_special_tokens=True)
    print(image_content.removeprefix(text))


def get_image_content(model: BlipForConditionalGeneration, processor: BlipProcessor) -> str:
    raw_image = get_image()

    # raw_image.show()

    # conditional image captioning
    text = "a photography of"

    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs, max_length=64)

    image_content = processor.decode(out[0], skip_special_tokens=True)
    return image_content.removeprefix(text)


if __name__ == "__main__":
    main()
