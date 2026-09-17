import vision_zoom, ocr, requests, json

els = ocr.get_elements('/tmp/test_zero_delay.png', 800)
els = [e for e in els if e['text'] != 'bon']
b64, mark_map = vision_zoom.prepare_zoom_som_view('/tmp/test_zero_delay.png', els, 1280, 800)

lines = []
for k, v in mark_map.items():
    lines.append("Mark [" + str(k) + "]: " + str(v.get('text')))
marks_text = "\n".join(lines)

prompt = f"""You control a computer desktop. Your current goal is to open mousepad.
Here is the zoomed view with numbered labels:
{marks_text}

Reply with raw JSON only: {{"action": "click_mark", "mark": <id>, "expected": "mousepad window opens"}}
"""

resp = requests.post('http://host.docker.internal:11434/api/generate', json={
    'model': 'qwen3-vl:latest',
    'prompt': prompt,
    'images': [b64],
    'stream': False,
    'format': 'json'
})
data = resp.json()
print("qwen3-vl decision:\n", data.get("response") or data.get("thinking"))
