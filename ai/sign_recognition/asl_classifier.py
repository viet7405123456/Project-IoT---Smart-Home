"""ASL alphabet classifier using the pretrained HuggingFace SigLIP model
https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection

This module is only imported when config.RECOGNITION_MODE == "asl_alphabet",
so `torch`/`transformers` are not required for the default finger-count mode.
"""

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

_model = None
_processor = None


def _load_model(model_name):
    global _model, _processor
    if _model is None:
        print(f"Loading ASL classifier '{model_name}' (first run downloads it)...")
        _processor = AutoImageProcessor.from_pretrained(model_name)
        _model = SiglipForImageClassification.from_pretrained(model_name)
        _model.eval()
    return _model, _processor


def crop_hand_region(frame, landmarks, margin_ratio):
    """landmarks: list of 21 MediaPipe NormalizedLandmark objects.
    Returns a cropped BGR image around the hand, with margin, or None if the
    box is degenerate.
    """
    height, width = frame.shape[:2]
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w <= 0 or box_h <= 0:
        return None

    margin_x = box_w * margin_ratio
    margin_y = box_h * margin_ratio

    x1 = max(0, int(min_x - margin_x))
    y1 = max(0, int(min_y - margin_y))
    x2 = min(width, int(max_x + margin_x))
    y2 = min(height, int(max_y + margin_y))

    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def classify(cropped_frame_bgr, model_name):
    """Returns (letter, confidence) for a cropped BGR hand image, e.g. ('L', 0.87)."""
    model, processor = _load_model(model_name)

    rgb_image = cv2.cvtColor(cropped_frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    best_idx = int(torch.argmax(probs).item())
    confidence = float(probs[best_idx].item())
    letter = model.config.id2label[best_idx].upper()
    return letter, confidence
