#!/bin/bash
set -e

Xvfb $DISPLAY -screen 0 $RESOLUTION &
sleep 1

fluxbox &
sleep 1

x11vnc -display $DISPLAY -forever -shared -nopw -rfbport 5900 -quiet &
sleep 1

websockify --web=/usr/share/novnc 6080 localhost:5900 &
sleep 1

echo "Sandbox ready. View it at http://localhost:6080/vnc.html"
echo "Waiting for task via TASK env var or /agent/task.txt ..."

# Keep container alive and run the agent loop if a task is provided
if [ -n "$TASK" ]; then
    python3 /agent/agent_sandbox.py "$TASK"
else
    # idle: just keep the display/VNC up so the user can watch or run tasks manually
    tail -f /dev/null
fi
