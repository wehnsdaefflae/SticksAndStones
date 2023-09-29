import requests
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering, AutoProcessor


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


def get_blip_conditional_model() -> BlipForConditionalGeneration:
    return BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


def get_blip_qna_model() -> BlipForConditionalGeneration:
    return BlipForQuestionAnswering.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")


def get_blip_conditional_processor() -> BlipProcessor:
    return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")


def get_blip_qna_processor() -> BlipProcessor:
    return AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")


def get_image_content(image: Image, model: BlipForConditionalGeneration, processor: BlipProcessor) -> str:
    text = "a photography of"

    inputs = processor(images=image, text=text, return_tensors="pt").to("cuda")

    out = model.generate(**inputs, max_length=64)

    image_content = processor.decode(out[0], skip_special_tokens=True)
    return image_content.removeprefix(text)


def main() -> None:
    raw_image = get_image()

    raw_image.show()

    processor = get_blip_conditional_processor()
    model = get_blip_conditional_model()

    image_content = get_image_content(raw_image, model, processor)

    print(image_content)


if __name__ == "__main__":
    main()
