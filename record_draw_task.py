#!/usr/bin/env python3
import record_task

TASK = "open drawing and draw a house on the canvas"
OUT = "/agent/drawing_demo.mp4"

if __name__ == "__main__":
    record_task.record_task(TASK, OUT)
