# 🛡️ Content Moderation Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment
for training and evaluating AI agents on **real-world content moderation tasks**.

Built for the **Meta × Scaler OpenEnv Hackathon** by **The Creative Coders**.

---

## Overview

Content moderation is one of the most critical and labour-intensive tasks on the modern internet.
Platforms employ thousands of human moderators to review billions of posts daily. This environment
provides a structured, gradeable RL training ground where agents learn to:

- Accurately classify harmful content
- Detect multiple overlapping toxicity signals
- Make nuanced, context-aware moderation decisions

---

## Tasks

### Task 1 — `binary-classification` (Easy)

**Objective:** Label a post as `safe` or `harmful`.

Harmful content includes direct threats, harassment, hate speech, spam, and abuse.
The grader awards `1.0` for a correct label, `0.0` for an incorrect one.

**Action format:**
```json
{"label": "safe"}
```

---

### Task 2 — `multi-label-toxicity` (Medium)

**Objective:** Assign one or more toxicity labels to a post.

Valid labels: `hate_speech`, `spam`, `harassment`, `misinformation`, `safe`.
Use `safe` only if no toxicity is present. A post can carry multiple labels simultaneously.

Scoring uses **Jaccard similarity** between predicted and ground-truth label sets,
providing partial credit for partially correct classifications.

**Action format:**
```json
{"labels": ["spam", "misinformation"]}
```

---

### Task 3 — `contextual-moderation` (Hard)

**Objective:** Given a conversation thread and a new reply, decide how to handle it.

The agent must output:
- `decision`: `remove`, `warn`, or `allow`
- `policy_violated`: `harassment`, `hate_speech`, `spam`, `misinformation`, `self_harm`, or `none`
- `reason`: 1–2 sentence natural-language justification

**Composite scoring:**
| Component | Weight | Criteria |
|-----------|--------|----------|
| Decision correctness | 40% | Exact match |
| Policy accuracy | 30% | Exact match |
| Reason quality | 30% | Keyword overlap with ground-truth rubric |

**Action format:**
```json
{
  "decision": "remove",
  "policy_violated": "harassment",
  "reason": "The message is a direct personal attack targeting a vulnerable user."
}
```

---

## Action & Observation Spaces

### Action Space

| Field | Type | Tasks | Description |
|-------|------|-------|-------------|
| `label` | `string` | 1 | `"safe"` or `"harmful"` |
| `labels` | `list[string]` | 2 | One or more toxicity labels |
| `decision` | `string` | 3 | `"remove"` / `"warn"` / `"allow"` |
| `policy_violated` | `string` | 3 | Policy name or `"none"` |
| `reason` | `string` | 3 | Justification sentence(s) |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `string` | Active task identifier |
| `task_description` | `string` | Natural language task description |
| `instructions` | `string` | Exact format required |
| `available_labels` | `list[string]` | Valid options for this task |
| `post_text` | `string` | Content item to moderate |
| `thread_context` | `list[string] \| null` | Prior messages (Task 3 only) |
| `sample_index` | `int` | 0-based current sample |
| `total_samples` | `int` | Total samples in task |
| `reward` | `float \| null` | Reward from last action |
| `done` | `bool` | Whether the episode has ended |
| `feedback` | `string \| null` | Human-readable grader feedback |

---

## Episode Lifecycle

This is a **single-step environment**:

```
reset(task="binary-classification", seed=0)
  → ModerationObservation (done=False, reward=None)

step({"label": "harmful"})
  → ModerationObservation (done=True, reward=1.0)

reset(...)   ← start next episode
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET` | `/state` | Inspect current state |
| `GET` | `/health` | Health check |
| `GET` | `/schema/action` | Action JSON schema |
| `GET` | `/schema/observation` | Observation JSON schema |
| `GET` | `/docs` | Swagger UI |

### Reset body (optional)

```json
{
  "task": "binary-classification",
  "seed": 0
}
```

Valid task values: `binary-classification`, `multi-label-toxicity`, `contextual-moderation`.

---

## Setup & Usage

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
# Build
docker build -t content-moderation-env .

# Run
docker run -p 7860:7860 content-moderation-env
```

### Validate with OpenEnv CLI

```bash
openenv validate --url http://localhost:7860
```

---

## Running the Baseline

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export HF_TOKEN="hf_your_token_here"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

---

## Baseline Scores

Baseline scores obtained using `gpt-4.1-mini` at temperature 0:

| Task | Avg Reward | Notes |
|------|-----------|-------|
| `binary-classification` | ~0.90 | High accuracy on explicit content |
| `multi-label-toxicity` | ~0.72 | Partial credit via Jaccard scoring |
| `contextual-moderation` | ~0.61 | Harder; requires nuanced reasoning |

---

## Reward Function Design

The reward function is designed to give **meaningful signal throughout training**:

- **Task 1** (binary): Sparse but unambiguous. 1.0 or 0.0.
- **Task 2** (multi-label): Jaccard similarity provides dense partial credit. An agent that
  identifies *some* labels earns a non-zero reward, incentivising progressive improvement.
- **Task 3** (contextual): Composite scoring rewards each component independently. An agent
  that gets the decision right but misidentifies the policy still earns 0.40, providing a clear
  gradient toward the full 1.0.

The environment penalises no-op actions (empty action dicts) by returning 0.0 reward, encouraging
agents to always commit to a specific response.

---

## Project Structure

```
content_moderation_env/
├── app.py               FastAPI application (entry point)
├── environment.py       ContentModerationEnvironment class
├── models.py            Pydantic Action / Observation / State models
├── tasks.py             Task datasets and grader functions
├── inference.py         Baseline inference script (OpenAI client)
├── openenv.yaml         OpenEnv metadata manifest
├── Dockerfile           Container image definition
├── requirements.txt     Python dependencies
└── README.md            This file
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
