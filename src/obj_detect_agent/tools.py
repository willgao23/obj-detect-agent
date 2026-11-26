from langchain.tools import tool, ToolRuntime
from .agent import Context
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from typing import Dict
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from .helpers import decode_b64, map_classes
from functools import lru_cache
import numpy as np

MODEL = "IDEA-Research/grounding-dino-tiny"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

@tool
def yolo_obb(classes: list[str], runtime: ToolRuntime[Context]) -> Results:
    """Detect objects using the DOTA-trained YOLO model.
        
    Args:
        classes: DOTA class names to detect
    """
    if not runtime.context.isStub:
        model = YOLO("yolo26s-obb.pt", task="detect")
        b64 = runtime.context.b64
        img = decode_b64(b64)
        res = model.predict(source=img, classes=map_classes(classes), verbose=False)
        return res
    else:
        return Results(np.zeros((10, 10, 3)), "mockImgPath.jpg", {0: "mock"})

@lru_cache(maxsize=1)
def get_dino():
    """Cached function to load the grounding dino model"""
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to(DEVICE)
    return processor, model

@tool 
def grounding_dino(classes: list[str], runtime: ToolRuntime[Context]) -> list[Dict]:
    """Open-vocabulary detection for arbitrary class names.
            
    Args:
        classes: Free-text class names to detect
    """
    if not runtime.context.isStub:
        processor, model = get_dino()
        b64 = runtime.context.b64
        img = decode_b64(b64)
        query = ". ".join(classes).lower().strip()
        inputs = processor(images=img, text=query, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[img.size[::-1]]
        )
        return results
    else:
        return [{"scores": torch.empty(1,), "boxes": torch.empty(4,), "labels": ["mock"], "text_labels": ["mock"]}]