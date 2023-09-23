import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photography of"
text = "Diving into the minutiae of the photograph before us, one can discern"
text = "With an almost obsessive attention to detail, the tableau unfolds as"

text = "The person in the image is wearing"
text = "The hair of the person in the image is"
text = "The clothes of the person in the image are"
text = "The accessoires of the person in the image are"
text = "The person in the image looks like"
inputs = processor(raw_image, text, return_tensors="pt")
out = model.generate(**inputs, max_length=64)
print(processor.decode(out[0], skip_special_tokens=True))

