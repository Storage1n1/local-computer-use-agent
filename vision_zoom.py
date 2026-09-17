"""
Vision Zoom & Set-of-Marks (SoM) Grounding Module for Local Computer-Use Agents.

Designed for small Vision-Language Models (e.g. qwen3-vl, minicpm-v4.6, gemma4).
Instead of forcing a small model to predict raw (x, y) pixels across a full 1080p
or 1280x800 desktop, this module:
1. Identifies UI bounding boxes (text elements via OCR or UI contours via CV).
2. Computes the Region of Interest (ROI) containing interactive elements.
3. Crops and upscales (super-resolution/bicubic) the target region.
4. Overlays high-contrast, numbered Set-of-Marks (SoM) badges [1], [2], [3]...
5. The small model only outputs the integer label id!
6. Automatically translates the selected label back to absolute desktop (cx, cy) coordinates.
"""

import os
import io
import base64
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Compatibility with both older Pillow and newer Pillow 10+
try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC

BADGE_COLORS = [
    (230, 25, 75),    # Red
    (60, 180, 75),    # Green
    (0, 130, 200),    # Blue
    (245, 130, 48),   # Orange
    (145, 30, 180),   # Purple
    (70, 240, 240),   # Cyan
    (240, 50, 230),   # Magenta
    (210, 245, 60),   # Lime
    (0, 128, 128),    # Teal
    (170, 110, 40),   # Brown
]


def detect_ui_contours(image_path: str, min_area: int = 400, max_area: int = 500000) -> List[Dict]:
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    _, thresh = cv2.threshold(grad, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    h_img, w_img = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if min_area <= area <= max_area and w < w_img * 0.95 and h < h_img * 0.95:
            boxes.append({
                "x": x, "y": y, "w": w, "h": h,
                "cx": x + w // 2,
                "cy": y + h // 2
            })
    return boxes


def compute_target_roi(elements: List[Dict], screen_w: int, screen_h: int, margin: int = 24) -> Tuple[int, int, int, int]:
    if not elements:
        return (0, 0, screen_w, screen_h)
    
    min_x = min(e["x"] for e in elements)
    min_y = min(e["y"] for e in elements)
    max_x = max(e["x"] + e["w"] for e in elements)
    max_y = max(e["y"] + e["h"] for e in elements)
    
    x1 = max(0, min_x - margin)
    y1 = max(0, min_y - margin)
    x2 = min(screen_w, max_x + margin)
    y2 = min(screen_h, max_y + margin)
    
    return (x1, y1, x2, y2)


def crop_and_upscale(
    image: Image.Image,
    roi: Tuple[int, int, int, int],
    scale_factor: float = 2.0
) -> Tuple[Image.Image, float]:
    x1, y1, x2, y2 = roi
    crop = image.crop((x1, y1, x2, y2))
    
    w = int((x2 - x1) * scale_factor)
    h = int((y2 - y1) * scale_factor)
    
    max_dim = 1200
    if max(w, h) > max_dim:
        scale_factor = max_dim / max(x2 - x1, y2 - y1)
        w = int((x2 - x1) * scale_factor)
        h = int((y2 - y1) * scale_factor)
    
    if scale_factor > 1.0:
        crop_upscaled = crop.resize((w, h), RESAMPLE_BICUBIC)
    else:
        crop_upscaled = crop
        
    return crop_upscaled, scale_factor


def annotate_set_of_marks(
    image: Image.Image,
    elements: List[Dict],
    roi: Tuple[int, int, int, int],
    scale_factor: float = 1.0
) -> Tuple[Image.Image, Dict[int, Dict]]:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    x1, y1, _, _ = roi
    mark_map = {}
    
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    for idx, el in enumerate(elements, start=1):
        color = BADGE_COLORS[(idx - 1) % len(BADGE_COLORS)]
        
        bx1 = int((el["x"] - x1) * scale_factor)
        by1 = int((el["y"] - y1) * scale_factor)
        bx2 = int((el["x"] + el["w"] - x1) * scale_factor)
        by2 = int((el["y"] + el["h"] - y1) * scale_factor)
        
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(annotated.width, bx2)
        by2 = min(annotated.height, by2)
        
        draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
        
        tag_text = str(idx)
        tag_w = 18 + (len(tag_text) - 1) * 8
        tag_h = 16
        tag_x1 = max(0, bx1 - tag_w)
        tag_y1 = max(0, by1)
        
        if tag_x1 == 0 and bx1 < tag_w:
            tag_x1 = bx1
            tag_y1 = max(0, by1 - tag_h)
            
        draw.rectangle([tag_x1, tag_y1, tag_x1 + tag_w, tag_y1 + tag_h], fill=color)
        draw.text((tag_x1 + 4, tag_y1 + 1), tag_text, fill="white", font=font)
        
        mark_map[idx] = {
            "label_id": idx,
            "text": el.get("text", ""),
            "orig_x": el["x"],
            "orig_y": el["y"],
            "orig_w": el["w"],
            "orig_h": el["h"],
            "cx": el["cx"],
            "cy": el["cy"]
        }
        
    return annotated, mark_map


def prepare_zoom_som_view(
    image_path: str,
    elements: List[Dict],
    screen_w: int,
    screen_h: int,
    focus_elements: Optional[List[Dict]] = None
) -> Tuple[str, Dict[int, Dict]]:
    raw_img = Image.open(image_path)
    target_elements = focus_elements if (focus_elements and len(focus_elements) > 0) else elements
    
    # If no interactive elements or only bottom bar elements, don't over-crop to a tiny bar
    # keep a minimum sensible crop area or full screen
    if not target_elements:
        roi = (0, 0, screen_w, screen_h)
    else:
        roi = compute_target_roi(target_elements, screen_w, screen_h, margin=24)
        
    # If ROI is tiny (e.g. less than 150px), expand it
    rx1, ry1, rx2, ry2 = roi
    if (rx2 - rx1) < 200 or (ry2 - ry1) < 150:
        center_x = (rx1 + rx2) // 2
        center_y = (ry1 + ry2) // 2
        rx1 = max(0, center_x - 150)
        ry1 = max(0, center_y - 100)
        rx2 = min(screen_w, center_x + 150)
        ry2 = min(screen_h, center_y + 100)
        roi = (rx1, ry1, rx2, ry2)

    cropped_upscaled, scale = crop_and_upscale(raw_img, roi, scale_factor=2.0)
    annotated_img, mark_map = annotate_set_of_marks(cropped_upscaled, target_elements, roi, scale_factor=scale)
    
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return b64, mark_map
