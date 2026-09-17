"""
File-based structured memory for the computer-use loop.

Instead of feeding a small model raw conversation history (expensive, and
small models don't reason well over long transcripts), we keep one compact
text log per run: for every step, what action was taken, what we expected
to happen, what actually happened (observed by a quick VLM reflection call),
and what the guidance is for the next step. The main decision prompt only
ever sees this compact log, not a wall of past turns.

On task completion, a summary is appended to a persistent cross-run file
(task_history.md) so a *future* run of the agent can look up how a similar
task went before.
"""
import json
import os
import time

STEP_LOG = "/agent/memory/step_log.jsonl"
TASK_HISTORY = "/agent/memory/task_history.md"


def _ensure_dir():
    os.makedirs(os.path.dirname(STEP_LOG), exist_ok=True)


def reset_step_log():
    _ensure_dir()
    open(STEP_LOG, "w").close()


def append_step(step, action, expected, observed, guidance_next):
    _ensure_dir()
    entry = {
        "step": step,
        "action": action,
        "expected": expected,
        "observed": observed,
        "guidance_next": guidance_next,
        "ts": time.time(),
    }
    with open(STEP_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def recent_steps_text(n=5):
    """Compact text summary of the last n steps, for the decision prompt."""
    if not os.path.exists(STEP_LOG):
        return "(no steps taken yet)"
    lines = []
    with open(STEP_LOG) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    for e in entries[-n:]:
        lines.append(
            f"step {e['step']}: did {e['action']} | expected: {e['expected']} | "
            f"observed: {e['observed']} | next guidance: {e['guidance_next']}"
        )
    return "\n".join(lines) if lines else "(no steps taken yet)"


def relevant_task_history(task, max_chars=1500):
    """Pull past attempts at a similar task from the cross-run history file."""
    if not os.path.exists(TASK_HISTORY):
        return ""
    with open(TASK_HISTORY) as f:
        content = f.read()
    if not content.strip():
        return ""
    # crude relevance: keep whole file if small, else just the tail (most recent)
    if len(content) <= max_chars:
        return content
    return content[-max_chars:]


def append_task_summary(task, outcome, steps_taken, summary):
    _ensure_dir()
    with open(TASK_HISTORY, "a") as f:
        f.write(f"\n## Task: {task}\n")
        f.write(f"- Outcome: {outcome}\n")
        f.write(f"- Steps taken: {steps_taken}\n")
        f.write(f"- Summary: {summary}\n")
