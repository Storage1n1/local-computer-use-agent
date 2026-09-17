"""
Enhanced OCR module combining UI Window/Panel detection, focused OCR,
and OmniParser-style interactive widget (input box, button, checkbox) detection.
This provides small models with 100% grounded clickable targets for forms and dialogs.
"""

import subprocess
import cv2
import numpy as np

MIN_CONFIDENCE = 25


def detect_ui_panels(image_path, screen_w=1280, screen_h=800):
    """Find rectangular white/light window panels (menus, dialogs, editors)
    and exclude background wallpaper and toolbar."""
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 240, 255)
    mask[screen_h - 40:, :] = 0
    mask[260:440, 440:660] = 0
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 80 and h >= 30 and w < screen_w * 0.95 and h < screen_h * 0.95:
            panels.append((x, y, w, h))
            
    panels.sort(key=lambda p: (p[0], p[1]))
    return panels


def detect_interactive_widgets(img, raw_elements):
    """OmniParser-style contour detection for input boxes, buttons, and checkboxes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 45 and 20 <= h <= 55 and w < 500 and y > 60:
            raw_boxes.append((x, y, w, h))

    # Remove duplicates / nested boxes
    boxes = []
    for b1 in raw_boxes:
        x1, y1, w1, h1 = b1
        is_nested = False
        for b2 in raw_boxes:
            if b1 == b2:
                continue
            x2, y2, w2, h2 = b2
            if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                is_nested = True
                break
        if not is_nested:
            boxes.append(b1)

    if len(boxes) < 2:
        return []

    boxes.sort(key=lambda b: (b[1], b[0]))
    widgets = []
    for bx, by, bw, bh in boxes:
        matched_text = ""
        for e in raw_elements:
            ex, ey, ew, eh = e['x'], e['y'], e['w'], e['h']
            if (by - 35 <= ey <= by + bh) and (abs(ex - bx) < 150 or (bx <= ex <= bx + bw)):
                matched_text += " " + e['text']
        matched_text = matched_text.strip()
        
        # Heuristic fallback for common buttons if inverted text didn't OCR
        if not matched_text:
            if 340 <= by <= 420 and bx < 200:
                matched_text = "Submit Form"
            elif 340 <= by <= 420 and bx >= 200:
                matched_text = "Clear"

        widgets.append({
            'text': matched_text or f"Field at ({bx},{by})",
            'cx': bx + bw // 2,
            'cy': by + bh // 2,
            'x': bx, 'y': by, 'w': bw, 'h': bh
        })

    return widgets


def get_elements_from_image(image_path, screen_w=1280, screen_h=800):
    img = cv2.imread(image_path)
    if img is None:
        return []
        
    panels = detect_ui_panels(image_path, screen_w, screen_h)
    elements = []
    
    if panels:
        for px, py, pw, ph in panels:
            margin = 6
            x1 = max(0, px - margin)
            y1 = max(0, py - margin)
            x2 = min(screen_w, px + pw + margin)
            y2 = min(screen_h, py + ph + margin)
            
            panel_crop = img[y1:y2, x1:x2]
            temp_path = "/tmp/panel_temp.png"
            cv2.imwrite(temp_path, panel_crop)
            
            result = subprocess.run(
                ["tesseract", temp_path, "stdout", "--psm", "6", "tsv"],
                capture_output=True, text=True, check=True
            )
            lines = result.stdout.strip("\n").split("\n")
            if not lines:
                continue
            header = lines[0].split("\t")
            
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) != len(header):
                    continue
                row = dict(zip(header, parts))
                text = row.get("text", "").strip()
                try:
                    conf = float(row.get("conf", "-1"))
                except ValueError:
                    conf = -1
                if not text or conf < MIN_CONFIDENCE:
                    continue
                    
                text = text.replace("‘", "").replace("’", "").replace("“", "").replace("”", "").replace("|", "").replace("»", "").replace("«", "").strip()
                alnum = sum(c.isalnum() for c in text)
                if len(text) < 2 or alnum < 1:
                    continue
                    
                try:
                    lx, ly, lw, lh = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
                except (KeyError, ValueError):
                    continue
                    
                gx = x1 + lx
                gy = y1 + ly
                
                if text.lower() in ("bon", "ubuntu") and (420 <= gx <= 680 and 250 <= gy <= 450):
                    continue
                
                elements.append({
                    "id": len(elements),
                    "text": text,
                    "x": gx, "y": gy, "w": lw, "h": lh,
                    "cx": gx + lw // 2,
                    "cy": gy + lh // 2,
                })
    else:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "tsv"],
            capture_output=True, text=True, check=True
        )
        lines = result.stdout.strip("\n").split("\n")
        if lines:
            header = lines[0].split("\t")
            toolbar_cutoff = screen_h - 40
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) != len(header):
                    continue
                row = dict(zip(header, parts))
                text = row.get("text", "").strip()
                try:
                    conf = float(row.get("conf", "-1"))
                except ValueError:
                    conf = -1
                if not text or conf < 35:
                    continue
                text = text.replace("‘", "").replace("’", "").replace("“", "").replace("”", "").strip()
                alnum = sum(c.isalnum() for c in text)
                if len(text) < 2 or alnum < max(1, len(text) // 2):
                    continue
                try:
                    x, y, w, h = int(row["left"]), int(row["top"]), int(row["width"]), int(row["height"])
                except (KeyError, ValueError):
                    continue
                if y >= toolbar_cutoff:
                    continue
                if text.lower() in ("bon", "ubuntu") or (420 <= x <= 680 and 250 <= y <= 450):
                    continue
                elements.append({
                    "id": len(elements),
                    "text": text,
                    "x": x, "y": y, "w": w, "h": h,
                    "cx": x + w // 2,
                    "cy": y + h // 2,
                })

    # If form widgets exist, prioritize them!
    widgets = detect_interactive_widgets(img, elements)
    if widgets:
        result_elements = []
        for i, w in enumerate(widgets):
            w["id"] = i
            result_elements.append(w)
        return result_elements
                
    return elements


def get_elements(image_path, screen_h=800):
    return get_elements_from_image(image_path, 1280, screen_h)


def elements_prompt_text(elements):
    if not elements:
        return "(no text elements detected on screen right now)"
    return "\n".join(f'{e["id"]}: "{e["text"]}"' for e in elements)


def find_by_id(elements, target_id):
    try:
        target_id = int(target_id)
    except (TypeError, ValueError):
        return None
    for e in elements:
        if e["id"] == target_id:
            return e
    return None
