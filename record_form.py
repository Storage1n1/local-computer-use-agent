#!/usr/bin/env python3
import record_task

TASK = "open form_app, fill in the registration form with Name Jane Doe, Email jane@example.com, City New York, check terms agreement, and submit the form"
OUT = "/agent/form_filling_demo.mp4"

if __name__ == "__main__":
    record_task.record_task(TASK, OUT)
