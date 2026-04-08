"""
Task definitions, datasets, and graders for the Content Moderation Environment.

Three tasks of increasing difficulty:
  1. binary-classification  (Easy)   – Label a post as 'safe' or 'harmful'
  2. multi-label-toxicity   (Medium) – Assign one or more toxicity labels to a post
  3. contextual-moderation  (Hard)   – Given thread context, decide remove/warn/allow
                                        with a policy justification
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Task 1 – Binary classification
# ---------------------------------------------------------------------------

BINARY_CASES: List[Dict[str, Any]] = [
    {
        "text": "Just finished reading an amazing novel – highly recommend it to everyone!",
        "label": "safe",
    },
    {
        "text": "I will hunt you down and make you pay for what you said.",
        "label": "harmful",
    },
    {
        "text": "Does anyone have a good recipe for homemade pasta?",
        "label": "safe",
    },
    {
        "text": "CLICK HERE NOW! Win $5,000 guaranteed! Limited time! Share your bank details!",
        "label": "harmful",
    },
    {
        "text": "Happy anniversary to my wonderful parents – 30 years together!",
        "label": "safe",
    },
    {
        "text": "All immigrants should be thrown out. They are destroying this country.",
        "label": "harmful",
    },
    {
        "text": "Looking for beginner tips on learning to play the ukulele.",
        "label": "safe",
    },
    {
        "text": "You are pathetic and worthless. Nobody will ever love you.",
        "label": "harmful",
    },
    {
        "text": "The sunset tonight was absolutely gorgeous – shared a photo above!",
        "label": "safe",
    },
    {
        "text": "Buy 10,000 fake followers for $3. DM me. Guaranteed delivery in 24h.",
        "label": "harmful",
    },
]

BINARY_LABELS = ["safe", "harmful"]


def grade_binary(action: Dict[str, Any], sample: Dict[str, Any]) -> float:
    """Return 1.0 for correct label, 0.0 otherwise."""
    predicted = (action.get("label") or "").strip().lower()
    expected = sample["label"].lower()
    return 1.0 if predicted == expected else 0.0


# ---------------------------------------------------------------------------
# Task 2 – Multi-label toxicity detection
# ---------------------------------------------------------------------------

MULTI_LABEL_CASES: List[Dict[str, Any]] = [
    {
        "text": "URGENT!! Forward this message to 10 friends within 5 minutes or bad luck will follow you for 7 years!!",
        "labels": {"spam", "misinformation"},
    },
    {
        "text": "Those people always lie and cheat. You cannot trust any of them. They don't belong here.",
        "labels": {"hate_speech", "harassment"},
    },
    {
        "text": "The new coffee shop on Main Street has amazing croissants – definitely worth a visit.",
        "labels": {"safe"},
    },
    {
        "text": "COVID vaccines contain microchips that track your location. Share before they delete this!",
        "labels": {"misinformation", "spam"},
    },
    {
        "text": "Hey loser, nobody at school actually likes you. We all laugh at you behind your back.",
        "labels": {"harassment"},
    },
    {
        "text": "Scientists have discovered a new species of deep-sea fish off the coast of New Zealand.",
        "labels": {"safe"},
    },
    {
        "text": "Kill all [slur]. They are subhuman. We need to cleanse this country.",
        "labels": {"hate_speech", "harassment"},
    },
    {
        "text": "Earn $500 a day working from home! No experience needed! Click now! Limited slots!",
        "labels": {"spam", "misinformation"},
    },
    {
        "text": "I just adopted a rescue dog named Biscuit – he is settling in beautifully.",
        "labels": {"safe"},
    },
    {
        "text": "Doctors don't want you to know this one weird cure for cancer. Buy our supplements now!",
        "labels": {"misinformation", "spam"},
    },
]

MULTI_LABELS_ALL = ["hate_speech", "spam", "harassment", "misinformation", "safe"]


def grade_multi_label(action: Dict[str, Any], sample: Dict[str, Any]) -> float:
    """Return Jaccard similarity between predicted and ground-truth label sets."""
    raw = action.get("labels") or []
    predicted: Set[str] = {lbl.strip().lower() for lbl in raw}
    expected: Set[str] = {lbl.lower() for lbl in sample["labels"]}

    if not predicted and not expected:
        return 1.0

    intersection = predicted & expected
    union = predicted | expected
    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)
    return round(jaccard, 4)


# ---------------------------------------------------------------------------
# Task 3 – Contextual moderation
# ---------------------------------------------------------------------------

CONTEXTUAL_CASES: List[Dict[str, Any]] = [
    {
        "thread": [
            "User_A: I've been struggling a lot lately and don't see a way forward.",
            "User_B: I'm so sorry to hear that. You're not alone.",
            "User_C: Have you considered talking to a professional counsellor?",
        ],
        "target": "User_D: Nobody cares about you. Just end it already.",
        "decision": "remove",
        "policy_violated": "self_harm",
        "reason_keywords": ["self-harm", "vulnerable", "crisis", "dangerous", "threatening"],
    },
    {
        "thread": [
            "User_A: Does anyone know a good plumber in the downtown area?",
            "User_B: I used ABC Plumbing last year – great service.",
        ],
        "target": "User_C: Check out XYZ Plumbers! Click here ➡ bit.ly/xyz-promo for 20% off!! HURRY!!",
        "decision": "warn",
        "policy_violated": "spam",
        "reason_keywords": ["promotional", "unsolicited", "link", "spam", "advertisement"],
    },
    {
        "thread": [
            "User_A: The new park renovation looks great!",
            "User_B: Agreed – the playground equipment is really nice.",
        ],
        "target": "User_C: I walked through there yesterday and the fountain area is lovely too.",
        "decision": "allow",
        "policy_violated": "none",
        "reason_keywords": ["positive", "constructive", "community", "relevant", "benign"],
    },
    {
        "thread": [
            "User_A: I can't believe they hired someone with those politics.",
            "User_B: I know, it's concerning.",
        ],
        "target": "User_C: All [ethnic group] are criminals. They should never be trusted in any workplace.",
        "decision": "remove",
        "policy_violated": "hate_speech",
        "reason_keywords": ["hate", "discriminatory", "ethnic", "stereotype", "slur", "racist"],
    },
    {
        "thread": [
            "User_A: My daughter just started her first job today!",
            "User_B: Congratulations to her!",
        ],
        "target": "User_C: That's wonderful – what field is she going into?",
        "decision": "allow",
        "policy_violated": "none",
        "reason_keywords": ["benign", "positive", "supportive", "harmless", "relevant"],
    },
    {
        "thread": [
            "User_A: I am so frustrated with the new company policy.",
            "User_B: Yeah it seems pretty unfair.",
        ],
        "target": "User_C: You two are useless idiots who should be fired. Stop complaining and get back to work.",
        "decision": "remove",
        "policy_violated": "harassment",
        "reason_keywords": ["insult", "personal attack", "hostile", "abusive", "harassment"],
    },
    {
        "thread": [
            "User_A: Has anyone read the report about the city budget?",
        ],
        "target": "User_B: The mayor is secretly embezzling millions – my cousin who works there told me. Share this!",
        "decision": "warn",
        "policy_violated": "misinformation",
        "reason_keywords": ["unverified", "rumour", "claim", "false", "spread", "misinformation"],
    },
    {
        "thread": [
            "User_A: Excited for the team meeting tomorrow!",
            "User_B: Me too, lots to discuss.",
        ],
        "target": "User_C: Here is the shared agenda doc I've put together: [internal link]",
        "decision": "allow",
        "policy_violated": "none",
        "reason_keywords": ["helpful", "relevant", "constructive", "collaborative", "appropriate"],
    },
]

CONTEXTUAL_DECISIONS = ["remove", "warn", "allow"]
CONTEXTUAL_POLICIES = ["harassment", "hate_speech", "spam", "misinformation", "self_harm", "none"]


def grade_contextual(action: Dict[str, Any], sample: Dict[str, Any]) -> float:
    """
    Composite score:
      - 0.40  correct moderation decision
      - 0.30  correct policy violated
      - 0.30  reason quality (keyword overlap, capped at 1.0)
    """
    score = 0.0

    # Decision correctness (0.40)
    predicted_decision = (action.get("decision") or "").strip().lower()
    if predicted_decision == sample["decision"]:
        score += 0.40

    # Policy correctness (0.30)
    predicted_policy = (action.get("policy_violated") or "").strip().lower()
    if predicted_policy == sample["policy_violated"]:
        score += 0.30

    # Reason keyword quality (0.30)
    reason_text = (action.get("reason") or "").lower()
    required_keywords = sample["reason_keywords"]
    if reason_text and required_keywords:
        matched = sum(1 for kw in required_keywords if kw in reason_text)
        keyword_score = min(1.0, matched / max(1, len(required_keywords)))
        score += 0.30 * keyword_score

    return round(score, 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict[str, Any]] = {
    "binary-classification": {
        "name": "binary-classification",
        "difficulty": "easy",
        "description": (
            "Classify a piece of user-generated content as 'safe' or 'harmful'. "
            "Harmful content includes threats, hate speech, spam, harassment, or abuse."
        ),
        "instructions": (
            "Respond with a ModerationAction where `label` is exactly 'safe' or 'harmful'."
        ),
        "available_labels": BINARY_LABELS,
        "cases": BINARY_CASES,
        "grader": grade_binary,
    },
    "multi-label-toxicity": {
        "name": "multi-label-toxicity",
        "difficulty": "medium",
        "description": (
            "Assign one or more toxicity labels to a piece of content. "
            "A post may have multiple labels or just 'safe'. "
            "Valid labels: hate_speech, spam, harassment, misinformation, safe."
        ),
        "instructions": (
            "Respond with a ModerationAction where `labels` is a list containing one or more "
            "of: ['hate_speech', 'spam', 'harassment', 'misinformation', 'safe']. "
            "Use 'safe' only if no toxicity is present."
        ),
        "available_labels": MULTI_LABELS_ALL,
        "cases": MULTI_LABEL_CASES,
        "grader": grade_multi_label,
    },
    "contextual-moderation": {
        "name": "contextual-moderation",
        "difficulty": "hard",
        "description": (
            "Given a conversation thread and a target reply, decide whether to 'remove', "
            "'warn', or 'allow' the reply. Provide the policy violated and a short reason. "
            "Use 'none' for policy_violated when the decision is 'allow'."
        ),
        "instructions": (
            "Respond with a ModerationAction where: "
            "`decision` is 'remove', 'warn', or 'allow'; "
            "`policy_violated` is one of 'harassment', 'hate_speech', 'spam', "
            "'misinformation', 'self_harm', or 'none'; "
            "`reason` is a 1–2 sentence explanation."
        ),
        "available_labels": CONTEXTUAL_DECISIONS,
        "cases": CONTEXTUAL_CASES,
        "grader": grade_contextual,
    },
}

DEFAULT_TASK = "binary-classification"
