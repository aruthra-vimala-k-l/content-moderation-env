"""
Baseline inference script for the Content Moderation Environment.
Placed at the root of the project as required by the hackathon spec.

Usage
-----
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4.1-mini"
    export HF_TOKEN="hf_..."
    export ENV_BASE_URL="http://localhost:7860"   # optional, defaults shown
    python inference.py

Output format (required by OpenEnv evaluation)
-----------------------------------------------
[START] task=<task_name> env=content-moderation-env model=<model_name>
[STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Tasks configuration
# ---------------------------------------------------------------------------
TASKS_CONFIG = [
    {"task": "binary-classification",  "seeds": list(range(10))},
    {"task": "multi-label-toxicity",   "seeds": list(range(10))},
    {"task": "contextual-moderation",  "seeds": list(range(8))},
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a content moderation AI assistant. "
    "You will receive a piece of user-generated content and must moderate it. "
    "Always respond with ONLY a valid JSON object. "
    "Do not include any explanation, markdown, or text outside the JSON object. "
    "Always include 'task_name' and 'sample_index' fields exactly as shown in the observation."
)


def build_user_prompt(obs: Dict[str, Any]) -> str:
    task = obs.get("task_name", "")
    desc = obs.get("task_description", "")
    instructions = obs.get("instructions", "")
    post = obs.get("post_text", "")
    thread = obs.get("thread_context") or []
    labels = obs.get("available_labels", [])
    s_idx = obs.get("sample_index", 0)
    total = obs.get("total_samples", 0)

    lines = [
        f"Task: {task}  (sample {s_idx + 1}/{total})",
        f"Description: {desc}",
        "",
        "Content to moderate:",
        f'  "{post}"',
    ]
    if thread:
        lines += ["", "Conversation thread (context):"]
        for msg in thread:
            lines.append(f"  {msg}")
    lines += [
        "",
        f"Available options: {labels}",
        "",
        instructions,
        "",
        "IMPORTANT: Your JSON response MUST include these two routing fields:",
        f'  "task_name": "{task}"',
        f'  "sample_index": {s_idx}',
        "",
        "Full response examples by task:",
        f'  binary-classification:  {{"task_name":"{task}","sample_index":{s_idx},"label":"safe"}}',
        f'  multi-label-toxicity:   {{"task_name":"{task}","sample_index":{s_idx},"labels":["spam"]}}',
        f'  contextual-moderation:  {{"task_name":"{task}","sample_index":{s_idx},'
        '"decision":"remove","policy_violated":"harassment","reason":"Direct personal attack."}}',
        "",
        "Respond with ONLY the JSON object for this content:",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task: str, seed: int) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    # Unwrap nested observation if present
    return data.get("observation", data)


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def parse_action(raw: str, task_name: str, sample_index: int) -> Dict[str, Any]:
    """Extract JSON from LLM response; inject routing fields if missing."""
    text = raw.strip()
    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("\n")
        text = "\n".join(parts[1:-1]) if len(parts) > 2 else text
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end])
            except json.JSONDecodeError:
                obj = {}
        else:
            obj = {}

    # Ensure routing fields are present and correct
    obj["task_name"] = task_name
    obj["sample_index"] = sample_index
    return obj


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_cfg: Dict[str, Any]) -> None:
    task_name = task_cfg["task"]
    seeds = task_cfg["seeds"]

    print(f"[START] task={task_name} env=content-moderation-env model={MODEL_NAME}")

    all_rewards: List[float] = []
    last_error: Optional[str] = None

    for step_num, seed in enumerate(seeds, start=1):
        error_msg: Optional[str] = None
        reward_val: float = 0.0
        done_val: bool = False
        action_str: str = "null"

        try:
            # 1. Reset environment - get observation
            obs = env_reset(task_name, seed)
            s_idx = obs.get("sample_index", seed)

            # 2. Build prompt and call LLM
            prompt = build_user_prompt(obs)
            raw_response = call_llm(prompt)

            # 3. Parse action (inject routing fields)
            action_dict = parse_action(raw_response, task_name, s_idx)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # 4. Step environment
            result = env_step(action_dict)
            reward_val = float(result.get("reward") or 0.0)
            done_val = bool(result.get("done", True))

        except Exception as exc:
            error_msg = str(exc).replace("\n", " ")[:200]
            last_error = error_msg

        all_rewards.append(reward_val)
        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward_val:.2f} done={str(done_val).lower()} "
            f"error={error_msg or 'null'}"
        )

    avg = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    success = avg >= 0.5 and last_error is None
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

    print(
        f"[END] success={str(success).lower()} steps={len(seeds)} "
        f"rewards={rewards_str}"
    )
    print(f"# Average reward for {task_name}: {avg:.4f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"# Content Moderation Environment - Baseline Inference")
    print(f"# Model : {MODEL_NAME}")
    print(f"# Env   : {ENV_BASE_URL}\n")

    # Wait for environment to be ready
    for attempt in range(15):
        try:
            resp = requests.get(f"{ENV_BASE_URL}/health", timeout=8)
            if resp.status_code == 200:
                print("# Environment is healthy. Starting inference.\n")
                break
        except Exception:
            pass
        print(f"# Waiting for environment... ({attempt + 1}/15)")
        time.sleep(5)
    else:
        print("# ERROR: Environment did not become ready. Exiting.")
        sys.exit(1)

    for task_cfg in TASKS_CONFIG:
        run_task(task_cfg)

    print("# All tasks complete.")


if __name__ == "__main__":
    main()
