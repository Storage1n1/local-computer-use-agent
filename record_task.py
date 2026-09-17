#!/usr/bin/env python3
import os
import signal
import subprocess
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def record_task(task_desc: str, output_mp4: str):
    env = os.environ.copy()
    env["DISPLAY"] = ":1"

    print(f"=== Starting Autonomous Recording for: '{task_desc}' ===")
    
    # 1. Clean desktop
    try:
        res = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", ".*"], capture_output=True, text=True, env=env)
        for wid in res.stdout.strip().split():
            r = subprocess.run(["xdotool", "getwindowname", wid], capture_output=True, text=True, env=env)
            title = r.stdout.strip()
            if title and title not in ("Desktop", "fluxbox", "x11vnc", "xterm", "Settings Toolbar"):
                subprocess.run(["xdotool", "windowclose", wid], env=env)
                time.sleep(0.2)
    except Exception:
        pass
    for app in ["drawing", "mtpaint", "mousepad"]:
        subprocess.run(["pkill", "-9", "-x", app], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "/usr/bin/form_app"], stderr=subprocess.DEVNULL)
    subprocess.run(["xdotool", "mousemove", "100", "100", "click", "1"], env=env, stderr=subprocess.DEVNULL)
    time.sleep(1.0)

    # 2. Start ffmpeg recording
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-video_size", "1280x800",
        "-framerate", "15",
        "-f", "x11grab",
        "-i", ":1.0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        output_mp4
    ]
    print(f"Launching ffmpeg to {output_mp4}...")
    rec_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5) # Let ffmpeg initialize

    try:
        # 3. Run autonomous agent
        agent_cmd = ["python3", "/agent/agent_sandbox.py", task_desc]
        print(f"Executing: {' '.join(agent_cmd)}")
        agent_proc = subprocess.run(agent_cmd, env=env)
        print(f"Agent finished with returncode: {agent_proc.returncode}")
    finally:
        # 4. Gracefully terminate ffmpeg
        time.sleep(2.0) # Buffer last few frames
        print("Finalizing video recording...")
        rec_proc.send_signal(signal.SIGINT)
        try:
            rec_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rec_proc.kill()
            rec_proc.wait()

    if os.path.exists(output_mp4):
        size_mb = os.path.getsize(output_mp4) / (1024 * 1024)
        print(f"SUCCESS: Video saved to {output_mp4} ({size_mb:.2f} MB)")
    else:
        print(f"ERROR: {output_mp4} was not generated!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: record_task.py '<task description>' /path/to/output.mp4")
        sys.exit(1)
    record_task(sys.argv[1], sys.argv[2])
