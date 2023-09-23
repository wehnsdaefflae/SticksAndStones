from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


def ask(question: str) -> str:
    # prepare image + question
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # text = "How many cats are there?"

    print(f"Question: {question}")

    #url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    #image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    image = Image.open("markwernsdorfer_mgr2.jpg").convert('RGB')

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    print(f"Answer: {answer}\n")
    return answer


def contains_any(response: str, terms: list) -> bool:
    return any(term in response for term in terms)


def clothing_details() -> dict:
    details = dict()

    attire_visible = ask("Is the main attire in the image visible?")
    if contains_any(attire_visible, ["yes", "yep", "yeah"]):

        attire = ask("What type of clothes are visible?")
        if contains_any(attire, ["fancy", "formal", "business"]):
            details['type_of_clothes'] = "fancy"
            details['material_of_clothes'] = ask("What material or fabric does the attire look like?")

            formal_visible = ask("Is there a suit or dress visible?")
            if contains_any(formal_visible, ["suit"]):
                details['clothing_item'] = "suit"
                details['number_of_pieces'] = ask("How many pieces does the suit have?")
                details['color_of_suit'] = ask("What is the color of the suit?")
                details['pattern_of_suit'] = ask("Does the suit have any patterns or designs?")

            else:
                details['clothing_item'] = "dress"
                details['length_of_dress'] = ask("How long is the dress?")
                details['color_of_dress'] = ask("What is the color of the dress?")
                details['pattern_of_dress'] = ask("Does the dress have any patterns or designs?")

        else:
            details['type_of_clothes'] = "everyday"
            details['material_of_clothes'] = ask("What material or fabric does the attire look like?")

            # ... Continue the rest for casual wear ...

    # Adding footwear details
    footwear_visible = ask("Is any footwear visible in the image?")
    if contains_any(footwear_visible, ["yes", "yep", "yeah"]):
        details['type_of_footwear'] = ask("What type of footwear is visible?")
        details['color_of_footwear'] = ask("What is the color of the footwear?")
        details['pattern_of_footwear'] = ask("Does the footwear have any patterns or designs?")

    # Adding jewelry and accessory details
    jewelry_visible = ask("Is any jewelry visible?")
    if contains_any(jewelry_visible, ["yes", "yep", "yeah"]):
        details['type_of_jewelry'] = ask("What type of jewelry is visible?")
        details['details_or_color_of_jewelry'] = ask("What is the color or special feature of the jewelry?")

    accessory_visible = ask("Is any accessory like a hat, scarf, or belt visible?")
    if contains_any(accessory_visible, ["yes", "yep", "yeah"]):
        details['type_of_accessory'] = ask("What type of accessory is visible?")
        details['details_or_color_of_accessory'] = ask("What is the color or special feature of the accessory?")

    bag_visible = ask("Is there a bag or handbag visible?")
    if contains_any(bag_visible, ["yes", "yep", "yeah"]):
        details['type_of_bag'] = ask("What type of bag is visible?")
        details['color_of_bag'] = ask("What is the color of the bag?")
        details['pattern_of_bag'] = ask("Does the bag have any patterns or designs?")

    fit_visible = ask("Can you discern the fit of the attire?")
    if contains_any(fit_visible, ["yes", "yep", "yeah"]):
        details['fit_of_clothes'] = ask("How does the attire fit? Snug, loose, or regular?")

    sleeve_visible = ask("Can you see the sleeves of the attire?")
    if contains_any(sleeve_visible, ["yes", "yep", "yeah"]):
        details['type_of_sleeves'] = ask("What type of sleeves does it have?")

    neckline_visible = ask("Can you determine the neckline of the attire?")
    if contains_any(neckline_visible, ["yes", "yep", "yeah"]):
        details['type_of_neckline'] = ask("What kind of neckline does it have?")

    embellishment_visible = ask("Are there any embellishments like sequins or beads on the attire?")
    if contains_any(embellishment_visible, ["yes", "yep", "yeah"]):
        details['embellishments'] = ask("Describe the embellishments.")

    pockets_visible = ask("Can you spot any pockets on the attire?")
    if contains_any(pockets_visible, ["yes", "yep", "yeah"]):
        details['type_of_pockets'] = ask("What type of pockets are visible?")

    layer_visible = ask("Are multiple layers, like a jacket or cardigan, visible over the main attire?")
    if contains_any(layer_visible, ["yes", "yep", "yeah"]):
        details['type_of_outer_layer'] = ask("Describe the outer layer.")

    hair_accessory_visible = ask("Are any hair accessories visible?")
    if contains_any(hair_accessory_visible, ["yes", "yep", "yeah"]):
        details['type_of_hair_accessory'] = ask("What hair accessory is visible?")

    watch_visible = ask("Is a watch visible?")
    if contains_any(watch_visible, ["yes", "yep", "yeah"]):
        details['type_of_watch'] = ask("Describe the watch.")

    return details


def main() -> None:
    details = clothing_details()
    print(details)
    # use the above json to generate a snarky description of someone's outfit.


if __name__ == '__main__':
    main()

