"""
Local computer-use agent: local Ollama VLM (minicpm-v4.6) grounds screenshots
into mouse/keyboard actions, pyautogui executes them, loop repeats.

Free, fully offline equivalent of Anthropic's computer-use demo, minus the
Docker/VNC scaffolding since we're already driving the real desktop.

Usage:
    python agent.py "open notepad and type hello world"
"""
import base64
import io
import json
import sys
import time

import pyautogui
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "minicpm-v4.6"
MAX_STEPS = 25
PAUSE_BETWEEN_STEPS = 1.0

pyautogui.FAILSAFE = True  # slam mouse to a screen corner to abort
SCREEN_W, SCREEN_H = pyautogui.size()

SYSTEM_PROMPT = """You control a Windows desktop by outputting exactly one JSON action per turn.

Screenshot resolution: {w}x{h}. Task: {task}

Reply with RAW JSON ONLY. No markdown, no ```, no <tool_call>, no explanation, no extra text before or after.
Your entire reply must start with {{ and end with }}.

Valid actions (pick exactly one):
{{"action": "click", "x": <int>, "y": <int>}}
{{"action": "double_click", "x": <int>, "y": <int>}}
{{"action": "right_click", "x": <int>, "y": <int>}}
{{"action": "type", "text": "<string>"}}
{{"action": "key", "keys": "<pyautogui key name, e.g. enter, esc, win, ctrl+s>"}}
{{"action": "scroll", "amount": <int, positive=up negative=down>}}
{{"action": "wait"}}
{{"action": "done", "summary": "<what was accomplished>"}}

Coordinates are absolute pixels on the screenshot, origin top-left.
To open an app that is not already visible on screen: first send {{"action": "key", "keys": "win"}} to open
the Start menu search box, then on the NEXT turn send {{"action": "type", "text": "<app name>"}}, then on the
turn after that send {{"action": "key", "keys": "enter"}}. Do this one step at a time, looking at each new
screenshot before deciding the next step. Do not right-click the desktop to open apps.
Only output "done" when the task is genuinely complete.

Example valid reply: {{"action": "click", "x": 500, "y": 300}}
"""


def screenshot_b64():
    img = pyautogui.screenshot()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ask_vlm(task, step_history):
    prompt = SYSTEM_PROMPT.format(w=SCREEN_W, h=SCREEN_H, task=task)
    if step_history:
        prompt += "\n\nActions taken so far:\n" + "\n".join(step_history)

    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "images": [screenshot_b64()],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.1, "num_predict": 600},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()
    return raw


def parse_action(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in model output: {raw!r}")
    return json.loads(raw[start:end + 1])


def execute(action):
    kind = action.get("action")
    if kind == "click":
        pyautogui.click(action["x"], action["y"])
    elif kind == "double_click":
        pyautogui.doubleClick(action["x"], action["y"])
    elif kind == "right_click":
        pyautogui.rightClick(action["x"], action["y"])
    elif kind == "type":
        pyautogui.write(action["text"], interval=0.02)
    elif kind == "key":
        keys = action["keys"].split("+")
        pyautogui.hotkey(*keys) if len(keys) > 1 else pyautogui.press(keys[0])
    elif kind == "scroll":
        pyautogui.scroll(action["amount"])
    elif kind == "wait":
        pass
    elif kind == "done":
        return True
    else:
        print(f"  (unknown action kind {kind!r}, skipping)")
    return False


def main():
    if len(sys.argv) < 2:
        print('Usage: python agent.py "task description"')
        sys.exit(1)
    task = " ".join(sys.argv[1:])
    print(f"Task: {task}")
    print(f"Screen: {SCREEN_W}x{SCREEN_H}  Model: {MODEL}")
    print("Move mouse to a screen corner at any time to abort (pyautogui failsafe).\n")

    history = []
    for step in range(1, MAX_STEPS + 1):
        print(f"--- step {step} ---")
        raw = ask_vlm(task, history)
        try:
            action = parse_action(raw)
        except ValueError as e:
            print(f"parse error: {e}\nraw: {raw}")
            history.append(f"step {step}: your last reply was not valid JSON and was ignored, reply with JSON ONLY")
            time.sleep(PAUSE_BETWEEN_STEPS)
            continue

        print(f"action: {action}")
        try:
            done = execute(action)
        except Exception as e:
            print(f"  execute error: {e}")
            history.append(f"step {step}: {action} -> FAILED ({e}), try a different action")
            time.sleep(PAUSE_BETWEEN_STEPS)
            continue
        history.append(f"step {step}: {action}")

        if done:
            print(f"\nDONE: {action.get('summary', '')}")
            return

        time.sleep(PAUSE_BETWEEN_STEPS)

    print("\nMax steps reached without completion.")


if __name__ == "__main__":
    main()
