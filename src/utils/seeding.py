from __future__ import annotations

import hashlib
import json
from typing import Any

SEED_SCHEME_VERSION = "sha256-v1"
LONGITUDINAL_DESIGN_VERSION = "independent_longitudinal_trajectories_v2"

def _stable_seed(namespace: str, *parts: Any) -> int:
    payload = json.dumps(
        [SEED_SCHEME_VERSION, namespace, *parts],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

def derive_trajectory_seed(base_seed: int, trajectory_idx: int) -> int:
    if trajectory_idx < 1:
        raise ValueError("trajectory_idx must be at least 1")
    return _stable_seed("trajectory", int(base_seed), int(trajectory_idx))

def derive_episode_seed(
    trajectory_seed: int,
    episode_position: int,
    scenario_identity: str,
) -> int:
    if episode_position < 1:
        raise ValueError("episode_position must be at least 1")
    return _stable_seed(
        "episode",
        int(trajectory_seed),
        int(episode_position),
        str(scenario_identity),
    )

def derive_call_seed(
    episode_seed: int,
    agent_id: str,
    round_idx: int,
    role: str,
) -> int:
    return _stable_seed(
        "call",
        int(episode_seed),
        str(agent_id),
        int(round_idx),
        str(role),
    )

def derive_action_regeneration_seed(
    call_seed: int,
    cell_attempt_idx: int,
    regeneration_attempt: int,
) -> int:
    if cell_attempt_idx < 1:
        raise ValueError("cell_attempt_idx must be at least 1")
    if regeneration_attempt < 1:
        raise ValueError("regeneration_attempt must be at least 1")
    return _stable_seed(
        "action-regeneration",
        int(call_seed),
        int(cell_attempt_idx),
        int(regeneration_attempt),
    )
