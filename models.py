"""
Typed Pydantic models for the Content Moderation Environment.

Design note
-----------
The OpenEnv HTTP simulation server creates a fresh environment instance per
request (stateless HTTP). To support this, ModerationAction includes routing
fields (task_name, sample_index) that the agent echoes back from the
observation it received during reset(). The step() handler uses these to
look up the correct sample and compute a reward without needing shared state.

- ModerationAction       - what the agent sends on each step
- ModerationObservation  - what the agent receives back
- ModerationState        - internal bookkeeping (accessible via /state)
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import Field
from openenv.core import Action, Observation, State


class ModerationAction(Action):
    """
    Action submitted by the agent.

    Routing fields (echo from observation)
    ----------------------------------------
    task_name    : str  - echo the task_name from the observation
    sample_index : int  - echo the sample_index from the observation

    Task 1 - binary-classification:
        Set `label` to "safe" or "harmful".

    Task 2 - multi-label-toxicity:
        Set `labels` to a list from
        ["hate_speech", "spam", "harassment", "misinformation", "safe"].

    Task 3 - contextual-moderation:
        Set `decision`         -> "remove" | "warn" | "allow"
        Set `policy_violated`  -> "harassment" | "hate_speech" | "spam" |
                                   "misinformation" | "self_harm" | "none"
        Set `reason`           -> 1-2 sentence explanation
    """
    # routing (agent echoes from observation)
    task_name: str = Field(
        description="Task name echoed from observation (e.g. 'binary-classification')"
    )
    sample_index: int = Field(
        description="Sample index echoed from observation"
    )
    # Task 1
    label: Optional[str] = Field(default=None, description="'safe' or 'harmful'")
    # Task 2
    labels: Optional[List[str]] = Field(
        default=None,
        description="List from ['hate_speech','spam','harassment','misinformation','safe']"
    )
    # Task 3
    decision: Optional[str] = Field(
        default=None, description="'remove' | 'warn' | 'allow'"
    )
    reason: Optional[str] = Field(
        default=None, description="1-2 sentence justification"
    )
    policy_violated: Optional[str] = Field(
        default=None,
        description="'harassment'|'hate_speech'|'spam'|'misinformation'|'self_harm'|'none'"
    )


class ModerationObservation(Observation):
    """Observation returned to the agent after reset() or step()."""
    task_name: str = Field(description="Current task identifier")
    task_description: str = Field(description="What the agent must accomplish")
    instructions: str = Field(description="Exact action format required")
    available_labels: List[str] = Field(description="Valid label/decision options")
    post_text: str = Field(description="Content item to moderate")
    thread_context: Optional[List[str]] = Field(
        default=None, description="Prior messages (Task 3 only)"
    )
    sample_index: int = Field(description="0-based index of the current sample")
    total_samples: int = Field(description="Total samples in this task")
    feedback: Optional[str] = Field(
        default=None, description="Grader feedback (populated after step)"
    )


class ModerationState(State):
    """Internal episode state - accessible via GET /state."""
    task_name: str = Field(default="", description="Active task name")
    sample_index: int = Field(default=0, description="Current sample index")
    total_samples: int = Field(default=0, description="Total samples in task")
    cumulative_reward: float = Field(default=0.0)
    step_rewards: List[float] = Field(default_factory=list)
