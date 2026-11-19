from langchain.tools import tool
from ultralytics import YOLO, YOLOE
from ultralytics.engine.results import Results
from PIL import Image

@tool
def yolo(img: Image, classes: list[int]) -> Results:
    """Tool call for YOLO detection"""
    model = YOLO("yolo26n.pt", task="detect")
    res = model.predict(source=img, classes=classes)
    return res

@tool
def yoloE(img: Image, classes: list[str]) -> Results:
    """Tool call for YOLOE detection with provided class names"""
    model = YOLOE("yoloe-26n-seg.pt", task="detect")
    model.set_classes(classes=classes)
    res = model.predict(source=img)
    return res

@tool
def yoloE_promptFree(img: Image) -> Results:
    """Tool call for prompt free YOLOE detection"""
    model = YOLOE("yoloe-26n-seg-pf.pt", task="detect")
    res = model.predict(source=img)
    return res