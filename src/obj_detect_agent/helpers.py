from io import BytesIO
import base64
from PIL import Image
from langchain.messages import HumanMessage
import yaml
import spacy

def map_classes(dota_classes: list[str]) -> list[int]:
    """A helper function to map string DOTA classes to their corresponding IDs"""
    with open("./data/dota.yaml") as fp:
        dict = yaml.safe_load(fp)['names']
    return [dict[dota_class] for dota_class in dota_classes]

def all_dota_class(args: list[str]) -> bool:
    """A helper function to determine if all of the arguments are DOTA classes"""
    with open("./data/dota.yaml") as fp:
        dict = yaml.safe_load(fp)['names']
        dota_classes = set(dict.keys())
        for arg in args:
            if arg not in dota_classes:
                return False
        return True

def map_tool_type(tool_call: str, classes: list[str]) -> str:
    """A helper function to map agent tool call and arguments to match the dataset type schema"""
    if (tool_call == "yolo_obb" and len(classes) == 0):
        return "general"
    elif (tool_call == "yolo_obb" and len(classes) > 0):
        return "dota"
    elif (tool_call == "grounding_dino"):
        return "open_vocab"
    else:
        return "N/A"

def find_object(prompt: str) -> list[str]:
    """A helper function which takes a doc a returns the first noun as the object to detect"""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(prompt)
    nouns = set()
    with open("./data/dota.yaml") as fp:
        dict = yaml.safe_load(fp)['names']
        dota_classes = set(dict.keys())
        nouns.update(filter(None, [dota_class if dota_class in doc.text else None for dota_class in dota_classes]))
    non_object_nouns = set(["boxes", "location", "coordinates"])
    general_nouns = set(["everything", "stuff", "all"])
    for chunk in doc.noun_chunks:
        if ((chunk.root.dep_ == "dobj") and 
            chunk.root.text not in non_object_nouns and
            chunk.root.text not in general_nouns):
            nouns.add(chunk.root.text)
    return list(nouns)
    
def convert_b64(path: str) -> str:
    """A helper function to convert an image in a path to base64"""
    with Image.open(path) as img:
        buf = BytesIO()
        img.save(buf, format="jpeg")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def decode_b64(b64: str) -> Image.Image:
    """A helper function to decode a base64 encoded image"""
    bytes = base64.b64decode(b64) 
    file = BytesIO(bytes)
    return Image.open(file)

def convert_msg(prompt: str, b64: str=None) -> HumanMessage:
    """A helper function to convert a prompt and a base64 encoded image into a Langchain compatible msg"""
    text_content = {"type": "text", "text": f"{prompt}"}
    img_content = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"}
    return HumanMessage(content=[text_content, img_content]) if b64 is not None else HumanMessage(content=[text_content])