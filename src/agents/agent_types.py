from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.utils.types import AGENT_TYPE_IDS

MALICIOUS_TYPES = ("malicious_strategic", "malicious_blocker")
HONEST_TYPES = ("hard_bargainer", "priority_purist", "general")

TYPE_DISPLAY_NAMES = {
    "malicious_blocker": "Malicious (Strategic)",
    "malicious_strategic": "Malicious (Strategic)",
    "hard_bargainer": "Honest (Hard Bargainer)",
    "priority_purist": "Honest (Priority Purist)",
    "general": "Honest (General)",
}

@dataclass
class TypeDesignBand:
    top_hi_lo: int
    top_hi_hi: int
    margin_lo: int
    margin_hi: int

TYPE_BANDS: dict[str, TypeDesignBand] = {
    "hard_bargainer": TypeDesignBand(26, 30, 0, 100),
    "priority_purist": TypeDesignBand(26, 30, 0, 100),
    "general": TypeDesignBand(24, 28, 0, 100),

    "malicious_blocker": TypeDesignBand(24, 30, 0, 100),
    "malicious_strategic": TypeDesignBand(24, 30, 0, 100),
}

@dataclass
class TypedAgentSpec:
    id: str
    name: str
    agent_type: str
    min_threshold: int
    detailed_goal_description: str
    priorities: list[tuple[str, str, int]]
    issue_scores: dict[str, dict[str, int]]
    issue_max_scores: dict[str, int]
    priority_narrative: str
    priorities_of_others: str

    priority_description: str = ""
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if self.agent_type == "malicious_blocker":
            self.agent_type = "malicious_strategic"

    @property
    def is_malicious(self) -> bool:
        return self.agent_type in MALICIOUS_TYPES

    def to_yaml_agent_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "min_threshold": self.min_threshold,
            "goal_description": self.detailed_goal_description,
            "priority_description": self.priority_description,
            "priority_narrative": self.priority_narrative,
            "issue_max_scores": dict(self.issue_max_scores),
            "scores": {k: dict(v) for k, v in self.issue_scores.items()},
            "preference_examples_about_others": self.priorities_of_others,
        }

def parse_composition(spec: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Bad composition chunk '{chunk}': expected '<type>:<count>'"
            )
        type_id, _, raw_count = chunk.partition(":")
        type_id = type_id.strip()
        if type_id not in AGENT_TYPE_IDS:
            raise ValueError(
                f"Unknown agent type '{type_id}'; must be one of {sorted(AGENT_TYPE_IDS)}"
            )
        if type_id == "malicious_blocker":
            type_id = "malicious_strategic"
        try:
            count = int(raw_count.strip())
        except ValueError as exc:
            raise ValueError(f"Bad count in composition chunk '{chunk}'") from exc
        if count < 0:
            raise ValueError(f"Negative count in composition chunk '{chunk}'")
        counts[type_id] = counts.get(type_id, 0) + count

    n_malicious = sum(counts.get(t, 0) for t in MALICIOUS_TYPES)
    n_honest = sum(counts.get(t, 0) for t in HONEST_TYPES)
    if n_malicious != 1:
        raise ValueError(
            f"Composition must contain exactly 1 malicious agent, got {n_malicious}"
        )
    if n_honest != 4:
        raise ValueError(
            f"Composition must contain exactly 4 honest agents, got {n_honest}"
        )
    return counts

def parse_assignment(
    spec: str,
    agent_ids: list[str],
    malicious_id: str,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Bad assignment chunk '{chunk}': expected '<agent_id>:<type>'"
            )
        aid, _, type_id = chunk.partition(":")
        aid, type_id = aid.strip(), type_id.strip()
        if aid not in agent_ids:
            raise ValueError(
                f"Unknown agent id '{aid}'; must be one of {agent_ids}"
            )
        if aid in mapping:
            raise ValueError(f"Duplicate assignment for agent '{aid}'")
        if type_id not in AGENT_TYPE_IDS:
            raise ValueError(
                f"Unknown agent type '{type_id}'; must be one of {sorted(AGENT_TYPE_IDS)}"
            )
        if type_id == "malicious_blocker":
            type_id = "malicious_strategic"
        mapping[aid] = type_id

    missing = [aid for aid in agent_ids if aid not in mapping]
    if missing:
        raise ValueError(f"Assignment is missing agents: {missing}")
    for aid, type_id in mapping.items():
        if type_id in MALICIOUS_TYPES and aid != malicious_id:
            raise ValueError(
                f"Malicious type '{type_id}' assigned to '{aid}', but the engine "
                f"runs '{malicious_id}' as the malicious agent (malicious.agent_id)"
            )
    if mapping[malicious_id] not in MALICIOUS_TYPES:
        raise ValueError(
            f"'{malicious_id}' is the engine's malicious slot and must get a "
            f"malicious type, got '{mapping[malicious_id]}'"
        )
    return mapping

def honest_type_sequence(counts: dict[str, int]) -> list[str]:
    order = ["hard_bargainer", "priority_purist", "general"]
    seq: list[str] = []
    for type_id in order:
        seq.extend([type_id] * counts.get(type_id, 0))
    return seq

def malicious_type(counts: dict[str, int]) -> str:
    for type_id in MALICIOUS_TYPES:
        if counts.get(type_id, 0):
            return "malicious_strategic"
    raise ValueError("Composition contains no malicious type")
