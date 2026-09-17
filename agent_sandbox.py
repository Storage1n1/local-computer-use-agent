"""
Enhanced Sandbox Agent with Vision Zoom, Super-Resolution Cropping, Set-of-Marks (SoM),
OS-level Grounding, Drawing Primitives, and Form Automation.
Synthesizes best practices from Agent-S, OmniParser, ShowUI, Browser-Use, and Anthropic Computer-Use.

Features:
1. Subdivided Milestones: breaks tasks into discrete subtask states.
2. Dual-Pass Grounding:
   - Identifies candidate UI elements (OCR + OmniParser-style contour box detection).
   - Zooms and upscales (super-resolution) the active region of interest (ROI).
   - Overlays numbered Set-of-Marks badges [1], [2], [3]...
3. Small-Model Safe:
   - Model NEVER predicts raw pixel coordinates for buttons or menus.
   - High-level primitives: {"action": "launch_app", "app": "form_app"}, {"action": "fill_field", "mark": <int>, "text": "..."}, {"action": "click_mark", "mark": <int>}, {"action": "draw_shape", "shape": "..."}, {"action": "type"}.
   - System maps marks back to absolute desktop (cx, cy) coordinates and handles smooth interpolated dragging for strokes.
4. OS-Level Grounding & Anti-Hallucination:
   - Queries real window manager state (active window titles) from X11.
   - Enforces grounding truth in decision making and visual reflection.
5. Problem Solving & Reflection:
   - Self-contained memory log (action -> expected -> observed -> guidance).
   - Detects loops and triggers diagnostic recovery.
"""

import base64
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

import memory
import ocr
import vision_zoom

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
MODEL = os.environ.get("MODEL", "qwen3-vl:latest")
MAX_STEPS = 40
PAUSE_BETWEEN_STEPS = 1.0
DISPLAY = os.environ.get("DISPLAY", ":1")

W, H = [int(x) for x in os.environ.get("RESOLUTION", "1280x800x24").split("x")[0:2]]
SHOT_PATH = "/tmp/shot.png"
ZOOM_SHOT_PATH = "/tmp/zoom_shot.png"

MAIN_CONTEXT = """MAIN CONTEXT (always true, does not change step to step):
- Environment: a bare Linux desktop, fluxbox window manager, inside an isolated sandbox. Resolution {w}x{h}.
- Installed apps: "form_app" (User Registration Form), "drawing" (painting & drawing canvas), "mousepad" (text editor), "xterm" (terminal).
- APP LAUNCHING:
  * You can launch any installed app directly using {{"action": "launch_app", "app": "form_app"}} (or "drawing", "mousepad", "xterm").
  * Alternatively, open the root menu via {{"action": "open_context_menu"}}, hover/click "Applications", and click the app name.
- FORM FILLING:
  * When a form is open, each input field and button has a numbered badge!
  * To fill a field directly, use {{"action": "fill_field", "mark": <id>, "text": "<value>"}}.
  * To toggle a checkbox or click a button (e.g. Submit), use {{"action": "click_mark", "mark": <id>}}.
- DRAWING TASKS:
  * When "drawing" is open, use {{"action": "draw_shape", "shape": "house" | "face" | "box" | "circle" | "sun" | "tree", "subtask_done": true}}.
- Set-of-Marks Labels:
  * You are provided with a high-resolution zoomed view of the active UI area with numbered colored labels [1], [2], [3]...
  * To interact with any labeled button, field, or menu item, reference its mark number!
  * Do not output raw pixel coordinates for buttons or menu items.
"""

PLAN_PROMPT = """You control a Linux desktop (inside an isolated sandbox) to accomplish a user task.
Installed apps: "form_app" (User Registration Form), "drawing" (drawing/painting tool), "mousepad" (text editor), "xterm" (terminal).

Break this task into 2-3 concrete, ORDERED goal states.
Keep it strictly to the core required states:
- State 1: Target application is open and active on desktop.
- Final state: The requested user goal is visibly completed on screen.
Do NOT plan unnecessary intermediate steps (such as adjusting brush size, tools, or colors) unless the user explicitly requested them.

Task: {task}

Reply with RAW JSON ONLY: {{"subtasks": ["first goal state", "second goal state"]}}
"""

DECIDE_PROMPT = """You control a Linux desktop (inside an isolated sandbox) by outputting exactly one JSON action per turn.

{main_context}

Overall task: {task}

Active open window(s) on desktop: {open_windows}

Your ONLY goal right now is this ONE subtask — do not try to jump ahead to later subtasks:
>>> CURRENT SUBTASK: {current_subtask} <<<

Subtasks already completed: {done_subtasks}
Upcoming subtasks: {upcoming_subtasks}

The attached image is a high-resolution crop & zoom of the active UI region with numbered Set-of-Marks labels.
Here is the index of marks visible:
{marks_text}

Reply with RAW JSON ONLY. Include an "expected" field describing what should happen.
Valid actions (pick exactly one):
{{"action": "launch_app", "app": "form_app" | "drawing" | "mousepad" | "xterm", "expected": "launches target application"}}
{{"action": "open_context_menu", "expected": "the fluxbox root menu appears"}}
{{"action": "fill_field", "mark": <integer mark id>, "text": "<text to enter>", "expected": "...", "subtask_done": true or false}}
{{"action": "click_mark", "mark": <integer mark id>, "expected": "...", "subtask_done": true or false}}
{{"action": "hover_mark", "mark": <integer mark id>, "expected": "...", "subtask_done": true or false}}
{{"action": "click_canvas", "expected": "focus the drawing canvas"}}
{{"action": "draw_shape", "shape": "house" | "face" | "box" | "circle" | "sun" | "tree", "expected": "draws the requested drawing on the canvas", "subtask_done": true}}
{{"action": "click_editor", "expected": "cursor positioned in text editor body"}}
{{"action": "type", "text": "<text to type>", "expected": "..."}}
{{"action": "key", "keys": "<key e.g. Return, ctrl+s>", "expected": "..."}}
{{"action": "wait", "expected": "..."}}
{{"action": "done", "summary": "<what was accomplished>"}}

Rules:
- IMPORTANT: If 'Active open window(s)' is (none, bare desktop), you MUST launch the app first (e.g. {{"action": "launch_app", "app": "form_app"}})!
- When filling form fields, use "fill_field" with the field's mark id and the text to type.
- To submit the form, look for the "Submit Form" button's mark id and use "click_mark".
- TASK COMPLETION: If a success banner (e.g. "Success: Form Submitted Successfully!") is visible, or if the requested drawing is visible on the canvas, output {{"action": "done", "summary": "Task completed successfully"}} immediately!
- Set "subtask_done": true ONLY if the current subtask is already visibly achieved on screen.

Log of recent steps:
{recent_steps}
"""

REFLECT_PROMPT = """{main_context}

You just took this action: {action}
Expected: {expected}
Current subtask: {current_subtask}

Active open window(s) on desktop: {open_windows}

Here is the new screen after the action, with current visible marks:
{marks_text}

CRITICAL TRUTH RULES:
- If 'Active open window(s)' is (none, bare desktop), NO application is open! Do NOT hallucinate that you see a form or drawing.
- If the application IS open, check whether the fields were filled or the form submission message is visibly rendered.

In 1-2 short sentences each:
1. OBSERVED: what actually changed?
2. GUIDANCE: what should the next action do?
3. SUBTASK_DONE: true or false (is {current_subtask} fully done?)

Reply with RAW JSON ONLY: {{"observed": "...", "guidance_next": "...", "subtask_done": true or false}}
"""

PROBLEM_SOLVE_PROMPT = """{main_context}

You are STUCK on subtask: {current_subtask}.
History of failed attempts:
{stuck_log}

Active open window(s) on desktop: {open_windows}

Visible marks:
{marks_text}

Diagnose why the previous actions failed and choose a concrete recovery action.
Reply with RAW JSON ONLY:
{{"diagnosis": "...", "new_strategy": "...", "recovery_action": {{"action": "...", ...}}}}
"""


def run(cmd):
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    subprocess.run(cmd, check=True, env=env)


def screenshot():
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY
    subprocess.run(["import", "-window", "root", SHOT_PATH], check=True, env=env)
    with open(SHOT_PATH, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_open_windows() -> List[str]:
    try:
        env = os.environ.copy()
        env["DISPLAY"] = DISPLAY
        res = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", ".*"], capture_output=True, text=True, env=env)
        names = []
        for wid in res.stdout.strip().split():
            r = subprocess.run(["xdotool", "getwindowname", wid], capture_output=True, text=True, env=env)
            n = r.stdout.strip()
            if n and n not in ("Desktop", "fluxbox", "x11vnc", "xterm", "Settings Toolbar"):
                names.append(n)
        return names
    except Exception:
        return []


def call_ollama(prompt, image_b64, num_predict=300):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [image_b64] if image_b64 else [],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.1, "num_predict": num_predict, "num_ctx": 4096},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    res = (data.get("response") or data.get("thinking") or "").strip()
    return res


def parse_json(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found: {raw!r}")
    return json.loads(raw[start:end + 1])


def smooth_drag(points: List[Tuple[int, int]], steps=10, delay=0.01):
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        run(["xdotool", "mousemove", str(x1), str(y1)])
        time.sleep(0.02)
        run(["xdotool", "mousedown", "1"])
        time.sleep(0.02)
        for s in range(1, steps + 1):
            x = int(x1 + (x2 - x1) * s / steps)
            y = int(y1 + (y2 - y1) * s / steps)
            run(["xdotool", "mousemove", str(x), str(y)])
            time.sleep(delay)
        run(["xdotool", "mouseup", "1"])
        time.sleep(0.02)


def execute_action(action: Dict, mark_map: Dict[int, Dict]):
    kind = action.get("action")
    if kind == "launch_app":
        app = action.get("app", "form_app").strip()
        env = os.environ.copy()
        env["DISPLAY"] = DISPLAY
        subprocess.Popen([app], env=env)
        time.sleep(2.0)
    elif kind == "open_context_menu":
        run(["xdotool", "mousemove", str(W // 5), str(H // 5), "click", "3"])
    elif kind == "fill_field":
        mark_id = action.get("mark") or action.get("id")
        try:
            mark_id = int(mark_id)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid mark id {mark_id!r}")
        target = mark_map.get(mark_id)
        if not target:
            raise ValueError(f"Mark [{mark_id}] is not in the current visible marks map ({list(mark_map.keys())})")
        cx, cy = target["cx"], target["cy"]
        run(["xdotool", "mousemove", str(cx), str(cy), "click", "1"])
        time.sleep(0.2)
        run(["xdotool", "key", "ctrl+a", "BackSpace"])
        time.sleep(0.1)
        run(["xdotool", "type", "--delay", "30", str(action.get("text", ""))])
    elif kind in ("click_mark", "hover_mark", "click_element", "hover_element"):
        mark_id = action.get("mark") or action.get("id")
        try:
            mark_id = int(mark_id)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid mark id {mark_id!r}")
        
        target = mark_map.get(mark_id)
        if not target:
            raise ValueError(f"Mark [{mark_id}] is not in the current visible marks map ({list(mark_map.keys())})")
            
        cx, cy = target["cx"], target["cy"]
        target_text = target.get("text", "").lower()
        
        if "application" in target_text:
            run(["xdotool", "mousemove", str(cx), str(cy)])
            time.sleep(0.4)
            run(["xdotool", "click", "1"])
            time.sleep(0.3)
        else:
            run(["xdotool", "mousemove", str(cx), str(cy)])
            if "click" in kind:
                run(["xdotool", "click", "1"])
    elif kind == "click_canvas":
        run(["xdotool", "mousemove", "450", "280", "click", "1"])
    elif kind == "drag":
        fx = int(action.get("from_x", 300))
        fy = int(action.get("from_y", 200))
        tx = int(action.get("to_x", 450))
        ty = int(action.get("to_y", 350))
        smooth_drag([(fx, fy), (tx, ty)])
    elif kind == "draw_stroke":
        pts = action.get("points") or [(300, 200), (450, 200)]
        smooth_drag(pts)
    elif kind == "draw_shape":
        shape = str(action.get("shape", "house")).lower()
        cx = int(action.get("x", 420))
        cy = int(action.get("y", 350))
        run(["xdotool", "mousemove", str(cx), str(cy), "click", "1"])
        time.sleep(0.2)
        if "house" in shape:
            smooth_drag([(cx-70, cy-30), (cx+70, cy-30), (cx+70, cy+80), (cx-70, cy+80), (cx-70, cy-30)])
            smooth_drag([(cx-70, cy-30), (cx, cy-100), (cx+70, cy-30)])
            smooth_drag([(cx-20, cy+80), (cx-20, cy+30), (cx+20, cy+30), (cx+20, cy+80)])
            smooth_drag([(cx-50, cy), (cx-30, cy), (cx-30, cy+20), (cx-50, cy+20), (cx-50, cy)])
        elif "face" in shape or "smiley" in shape:
            circle_pts = [(int(cx + 60 * math.cos(math.radians(a))), int(cy + 60 * math.sin(math.radians(a)))) for a in range(0, 361, 20)]
            smooth_drag(circle_pts)
            smooth_drag([(cx-25, cy-20), (cx-25, cy-15)])
            smooth_drag([(cx+25, cy-20), (cx+25, cy-15)])
            smile_pts = [(int(cx + 35 * math.cos(math.radians(a))), int(cy + 10 + 35 * math.sin(math.radians(a)))) for a in range(30, 151, 15)]
            smooth_drag(smile_pts)
        elif "circle" in shape:
            r = int(action.get("size", 60))
            pts = [(int(cx + r * math.cos(math.radians(a))), int(cy + r * math.sin(math.radians(a)))) for a in range(0, 361, 15)]
            smooth_drag(pts)
        elif "sun" in shape:
            r = int(action.get("size", 40))
            pts = [(int(cx + r * math.cos(math.radians(a))), int(cy + r * math.sin(math.radians(a)))) for a in range(0, 361, 15)]
            smooth_drag(pts)
            for a in range(0, 360, 45):
                x1 = int(cx + (r + 5) * math.cos(math.radians(a)))
                y1 = int(cy + (r + 5) * math.sin(math.radians(a)))
                x2 = int(cx + (r + 30) * math.cos(math.radians(a)))
                y2 = int(cy + (r + 30) * math.sin(math.radians(a)))
                smooth_drag([(x1, y1), (x2, y2)])
        elif "tree" in shape:
            smooth_drag([(cx-15, cy+80), (cx-15, cy+10), (cx+15, cy+10), (cx+15, cy+80)])
            pts = [(int(cx + 50 * math.cos(math.radians(a))), int(cy - 20 + 50 * math.sin(math.radians(a)))) for a in range(0, 361, 15)]
            smooth_drag(pts)
        else:
            s = int(action.get("size", 80))
            smooth_drag([(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s), (cx-s, cy-s)])
    elif kind == "click_editor":
        run(["xdotool", "mousemove", "250", "250", "click", "1"])
    elif kind == "type":
        run(["xdotool", "type", "--delay", "30", action["text"]])
    elif kind == "key":
        run(["xdotool", "key", action["keys"]])
    elif kind == "scroll":
        button = "4" if action["amount"] > 0 else "5"
        run(["xdotool", "click", "--repeat", str(abs(action["amount"])), button])
    elif kind == "wait":
        time.sleep(1.0)
    elif kind == "done":
        return True
    else:
        print(f"  (unknown action {kind!r}, skipping)")
    return False


def get_marks_summary(mark_map: Dict[int, Dict]) -> str:
    if not mark_map:
        return "(no interactive UI text marks detected on screen right now)"
    lines = []
    for k, v in mark_map.items():
        lines.append(f"Mark [{k}]: \"{v.get('text', '')}\"")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print('Usage: agent_sandbox.py "task description"')
        sys.exit(1)
    task = " ".join(sys.argv[1:])
    print(f"Task: {task}")
    print(f"Sandbox display {DISPLAY} {W}x{H}  Model: {MODEL}", flush=True)

    memory.reset_step_log()
    last_action_sig = None
    repeat_count = 0

    full_shot = screenshot()
    plan_raw = call_ollama(PLAN_PROMPT.format(task=task), full_shot, num_predict=300)
    try:
        subtasks = parse_json(plan_raw).get("subtasks") or [task]
    except ValueError:
        subtasks = [task]
        
    print(f"Plan ({len(subtasks)} subtasks):", flush=True)
    for i, s in enumerate(subtasks):
        print(f"  {i + 1}. {s}", flush=True)
    memory.append_step(0, "(plan)", "-", "-", f"plan: {subtasks}")

    subtask_idx = 0
    steps_since_progress = 0
    STUCK_REPEAT_THRESHOLD = 3
    STUCK_NO_PROGRESS_THRESHOLD = 6

    for step in range(1, MAX_STEPS + 1):
        if subtask_idx >= len(subtasks):
            subtask_idx = len(subtasks) - 1
        current_subtask = subtasks[subtask_idx]
        done_subtasks = subtasks[:subtask_idx] or "(none yet)"
        upcoming_subtasks = subtasks[subtask_idx + 1:] or "(none, this is the last one)"
        is_last_subtask = subtask_idx == len(subtasks) - 1

        print(f"--- step {step} (subtask {subtask_idx + 1}/{len(subtasks)}: {current_subtask}) ---", flush=True)
        
        screenshot()
        elements = ocr.get_elements(SHOT_PATH, H)
        open_windows = get_open_windows() or ["(none, bare desktop)"]
        
        # Build Zoom & Set-of-Marks view
        zoom_b64, mark_map = vision_zoom.prepare_zoom_som_view(
            SHOT_PATH, elements, W, H
        )
        marks_text = get_marks_summary(mark_map)
        recent_steps = memory.recent_steps_text()

        stuck = repeat_count >= STUCK_REPEAT_THRESHOLD or steps_since_progress >= STUCK_NO_PROGRESS_THRESHOLD
        if stuck:
            print(f"  [problem-solving triggered: repeat_count={repeat_count}, steps_since_progress={steps_since_progress}]", flush=True)
            ps_raw = call_ollama(
                PROBLEM_SOLVE_PROMPT.format(
                    main_context=MAIN_CONTEXT.format(w=W, h=H),
                    current_subtask=current_subtask,
                    stuck_log=memory.recent_steps_text(n=8),
                    open_windows=open_windows,
                    marks_text=marks_text,
                ),
                zoom_b64, num_predict=350,
            )
            try:
                ps = parse_json(ps_raw)
                diagnosis = ps.get("diagnosis", "(no diagnosis)")
                new_strategy = ps.get("new_strategy", "(no strategy)")
                action = ps.get("recovery_action") or {"action": "launch_app", "app": "form_app", "expected": "recover by launching app"}
                print(f"  diagnosis: {diagnosis}", flush=True)
                print(f"  new strategy: {new_strategy}", flush=True)
                memory.append_step(step, "(problem-solving)", "-", f"diagnosis: {diagnosis}", f"strategy: {new_strategy}")
            except ValueError:
                action = {"action": "launch_app", "app": "form_app", "expected": "recover by launching app"}
            repeat_count = 0
            steps_since_progress = 0
        else:
            prompt = DECIDE_PROMPT.format(
                main_context=MAIN_CONTEXT.format(w=W, h=H),
                task=task,
                open_windows=open_windows,
                current_subtask=current_subtask,
                done_subtasks=done_subtasks,
                upcoming_subtasks=upcoming_subtasks,
                marks_text=marks_text,
                recent_steps=recent_steps,
            )
            raw = call_ollama(prompt, zoom_b64, num_predict=300)
            try:
                action = parse_json(raw)
            except ValueError as e:
                print(f"parse error: {e}", flush=True)
                memory.append_step(step, "(invalid JSON)", "-", "model reply was not valid JSON", "reply with valid JSON only")
                time.sleep(PAUSE_BETWEEN_STEPS)
                continue

        print(f"action: {action}", flush=True)
        expected = action.get("expected", action.get("summary", "task complete"))

        # Grounding sanity check: if model tries to click mark when no marks exist, convert to launch_app if app not open
        if action.get("action") in ("click_mark", "hover_mark", "fill_field") and not mark_map and not get_open_windows():
            target_app = "form_app" if "form" in task.lower() else "drawing"
            print(f"  (sanity: no marks and desktop empty, converting to launch_app {target_app})", flush=True)
            action = {"action": "launch_app", "app": target_app, "expected": f"launch {target_app} application"}

        if action.get("subtask_done") and subtask_idx < len(subtasks) - 1:
            subtask_idx += 1
            current_subtask = subtasks[subtask_idx]
            is_last_subtask = (subtask_idx == len(subtasks) - 1)
            print(f"  >>> subtask {subtask_idx} complete (self-reported), advancing to: {current_subtask}", flush=True)
            steps_since_progress = 0

        action_sig = (action.get("action"), action.get("mark"), action.get("text"), action.get("keys"), action.get("shape"), action.get("app"))
        if action_sig == last_action_sig:
            repeat_count += 1
        else:
            repeat_count = 0
        last_action_sig = action_sig

        if action.get("action") == "done":
            memory.append_task_summary(task, "completed", step, expected)
            print(f"\nDONE: {expected}", flush=True)
            return

        try:
            execute_action(action, mark_map)
        except Exception as e:
            print(f"  execute error: {e}", flush=True)
            memory.append_step(step, action, expected, f"action failed: {e}", "pick valid mark id from current image")
            time.sleep(PAUSE_BETWEEN_STEPS)
            continue

        if action.get("action") == "hover_mark":
            memory.append_step(step, action, expected, "(hover complete, acting immediately on fresh menu)", "click target app now")
            steps_since_progress += 1
            time.sleep(0.3)
            continue

        time.sleep(PAUSE_BETWEEN_STEPS)

        # Reflection
        screenshot()
        after_elements = ocr.get_elements(SHOT_PATH, H)
        after_open_windows = get_open_windows() or ["(none, bare desktop)"]
        after_zoom_b64, after_marks = vision_zoom.prepare_zoom_som_view(SHOT_PATH, after_elements, W, H)
        
        reflect_raw = call_ollama(
            REFLECT_PROMPT.format(
                main_context=MAIN_CONTEXT.format(w=W, h=H),
                action=action,
                expected=expected,
                current_subtask=current_subtask,
                open_windows=after_open_windows,
                marks_text=get_marks_summary(after_marks),
            ),
            after_zoom_b64, num_predict=250,
        )
        try:
            reflection = parse_json(reflect_raw)
            observed = reflection.get("observed", "(no observation)")
            guidance = reflection.get("guidance_next", "(no guidance)")
            subtask_done = bool(reflection.get("subtask_done", False))
        except ValueError:
            observed = "(reflection invalid JSON)"
            guidance = "proceed with care"
            subtask_done = False

        print(f"  observed: {observed}", flush=True)
        print(f"  guidance: {guidance}", flush=True)
        
        if subtask_done and subtask_idx < len(subtasks) - 1:
            subtask_idx += 1
            current_subtask = subtasks[subtask_idx]
            is_last_subtask = (subtask_idx == len(subtasks) - 1)
            print(f"  >>> subtask {subtask_idx} complete, advancing to: {current_subtask}", flush=True)
            steps_since_progress = 0
        else:
            steps_since_progress += 1
            
        memory.append_step(step, action, expected, observed, guidance)

    print("\nMax steps reached.", flush=True)
    memory.append_task_summary(task, "max steps reached", MAX_STEPS, memory.recent_steps_text(n=3))


if __name__ == "__main__":
    main()
