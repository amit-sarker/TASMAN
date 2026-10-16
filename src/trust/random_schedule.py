from __future__ import annotations

import hashlib
import json
from typing import Any

RANDOM_TRUST_PROTOCOL = "random_uniform_per_episode_v1"
_SAMPLER = "sha256_first_53_bits_v1"
_GRID_SIZE = 2**53

class RandomTrustSchedule:
    def __init__(self, agent_ids: list[str], base_seed: int) -> None:
        if not isinstance(agent_ids, list) or not agent_ids:
            raise ValueError("Random trust requires a nonempty list of agent IDs")
        if any(not isinstance(agent, str) or not agent.strip() for agent in agent_ids):
            raise ValueError("Random trust agent IDs must be nonempty strings")
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("Random trust agent IDs must be unique")
        if type(base_seed) is not int:
            raise ValueError("Random trust base_seed must be an integer")
        self._agent_ids = tuple(sorted(agent_ids))
        self._base_seed = base_seed

    @staticmethod
    def _validate_identity(trajectory_idx: int, episode_position: int) -> None:
        for name, value in (
            ("trajectory_idx", trajectory_idx),
            ("episode_position", episode_position),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"Random trust {name} must be a positive integer")

    def _draw(self, trajectory_idx: int, episode_position: int, observer: str, target: str) -> float:
        key = json.dumps(
            [RANDOM_TRUST_PROTOCOL, self._base_seed, trajectory_idx, episode_position, observer, target],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        numerator = int.from_bytes(hashlib.sha256(key).digest()[:7], "big") >> 3
        return numerator / _GRID_SIZE

    def snapshot(self, trajectory_idx: int, episode_position: int) -> dict[str, dict[str, float]]:
        self._validate_identity(trajectory_idx, episode_position)
        return {
            observer: {
                target: self._draw(trajectory_idx, episode_position, observer, target)
                for target in self._agent_ids
                if target != observer
            }
            for observer in self._agent_ids
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": RANDOM_TRUST_PROTOCOL,
            "sampler": _SAMPLER,
            "distribution": "uniform",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
            "upper_exclusive": True,
            "refresh": "per_episode",
            "independent_directed_pairs": True,
            "assessment_enabled": False,
            "base_seed": self._base_seed,
            "agent_ids": list(self._agent_ids),
        }

    def provenance(self, trajectory_idx: int, episode_position: int) -> dict[str, Any]:
        self._validate_identity(trajectory_idx, episode_position)
        return {
            **self.metadata(),
            "trajectory_idx": trajectory_idx,
            "episode_position": episode_position,
        }
