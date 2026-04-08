"""
ContentModerationEnvironment - the core OpenEnv environment.

Stateless HTTP design
---------------------
Because the OpenEnv HTTP simulation server creates a fresh environment
instance per request, this environment is designed to be stateless across
HTTP calls:

  reset(task, seed) -> picks a sample, returns ModerationObservation
                       (no persistent state needed)

  step(action)      -> action carries task_name + sample_index routing fields,
                       so the grader can look up the right sample without
                       needing shared state from a prior reset().

The state property reflects the last operation performed in this instance.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from openenv.core import Environment

from models import ModerationAction, ModerationObservation, ModerationState
from tasks import TASKS, DEFAULT_TASK


class ContentModerationEnvironment(
    Environment[ModerationAction, ModerationObservation, ModerationState]
):
    """
    RL environment for content moderation.

    Tasks
    -----
    binary-classification  (easy)   - label a post safe/harmful
    multi-label-toxicity   (medium) - assign multiple toxicity labels
    contextual-moderation  (hard)   - decide remove/warn/allow given thread context
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = ModerationState()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        """
        Start a new episode by selecting a sample.

        Parameters
        ----------
        seed : int, optional
            Selects sample at index (seed % len(cases)).
            When None, defaults to sample 0.
        task : str, optional
            One of 'binary-classification', 'multi-label-toxicity',
            'contextual-moderation'. Defaults to 'binary-classification'.
        episode_id : str, optional
            Custom episode identifier; auto-generated if not supplied.
        """
        task_name = (task or DEFAULT_TASK).strip()
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Valid tasks: {sorted(TASKS.keys())}"
            )

        task_cfg = TASKS[task_name]
        cases = task_cfg["cases"]
        idx = (seed % len(cases)) if seed is not None else 0

        self._state = ModerationState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            sample_index=idx,
            total_samples=len(cases),
        )

        return self._build_obs_from_idx(task_name, idx, reward=None, done=False, feedback=None)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: ModerationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ModerationObservation:
        """
        Evaluate the agent's moderation action and return reward.

        The action must include task_name and sample_index (echoed from the
        observation returned by reset()). This makes the step stateless and
        compatible with the HTTP simulation server.
        """
        task_name = action.task_name.strip()
        if task_name not in TASKS:
            raise ValueError(f"Unknown task_name in action: '{task_name}'")

        task_cfg = TASKS[task_name]
        cases = task_cfg["cases"]
        idx = action.sample_index % len(cases)
        sample = cases[idx]

        # Grade the action
        action_dict = action.model_dump(exclude={"metadata"})
        reward: float = task_cfg["grader"](action_dict, sample)

        # Update state for /state endpoint
        self._state.task_name = task_name
        self._state.sample_index = idx
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.step_rewards.append(reward)

        feedback = _generate_feedback(task_name, action_dict, sample, reward)

        return self._build_obs_from_idx(
            task_name, idx, reward=reward, done=True, feedback=feedback
        )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> ModerationState:
        return self._state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_obs_from_idx(
        self,
        task_name: str,
        idx: int,
        reward: Optional[float],
        done: bool,
        feedback: Optional[str],
    ) -> ModerationObservation:
        task_cfg = TASKS[task_name]
        cases = task_cfg["cases"]
        sample = cases[idx % len(cases)]

        return ModerationObservation(
            done=done,
            reward=reward,
            task_name=task_name,
            task_description=task_cfg["description"],
            instructions=task_cfg["instructions"],
            available_labels=task_cfg["available_labels"],
            post_text=sample.get("text") or sample.get("target", ""),
            thread_context=sample.get("thread"),
            sample_index=idx,
            total_samples=len(cases),
            feedback=feedback,
        )


# ------------------------------------------------------------------
# Feedback helpers (module-level, reused by tests)
# ------------------------------------------------------------------

def _generate_feedback(
    task_name: str,
    action: dict,
    sample: dict,
    reward: float,
) -> str:
    if task_name == "binary-classification":
        pred = (action.get("label") or "").lower()
        exp = sample["label"]
        if reward == 1.0:
            return f"Correct! The content was '{exp}'."
        return f"Incorrect. You predicted '{pred}', expected '{exp}'."

    if task_name == "multi-label-toxicity":
        pred = sorted(lbl.lower() for lbl in (action.get("labels") or []))
        exp = sorted(lbl.lower() for lbl in sample["labels"])
        return (
            f"Jaccard score: {reward:.2f}. "
            f"Your labels: {pred}. Expected: {exp}."
        )

    if task_name == "contextual-moderation":
        dec_p = (action.get("decision") or "").lower()
        dec_e = sample["decision"]
        pol_p = (action.get("policy_violated") or "").lower()
        pol_e = sample["policy_violated"]
        return (
            f"Decision: {'correct' if dec_p == dec_e else 'wrong'} "
            f"(got '{dec_p}', expected '{dec_e}'). "
            f"Policy: {'correct' if pol_p == pol_e else 'wrong'} "
            f"(got '{pol_p}', expected '{pol_e}'). "
            f"Total: {reward:.2f}/1.00."
        )

    return f"Score: {reward:.2f}"
