from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from src.agents.honest_agent import THRESHOLD_ACTION_POLICY
from src.agents.prompt_adapter import SUPPLEMENT_PROMPT_PROTOCOL
from src.utils.seeding import LONGITUDINAL_DESIGN_VERSION, SEED_SCHEME_VERSION
from src.utils.types import GameConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_FILENAME = "trust_replay.json"
REPLAY_PROTOCOL = "recorded_trust_replay_v1"

class TrustReplayError(ValueError):
    pass

def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()

def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrustReplayError(f"Duplicate JSON field in replay source: {key}")
        result[key] = value
    return result

def _invalid_constant(value: str) -> None:
    raise TrustReplayError(f"Non-finite JSON value in replay source: {value}")

def _parse(text: str, location: str) -> dict[str, Any]:
    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, TypeError) as exc:
        raise TrustReplayError(f"Invalid replay JSON at {location}: {exc}") from exc
    if not isinstance(result, dict):
        raise TrustReplayError(f"Expected a JSON object at {location}")
    return result

def _read_object(path: Path) -> dict[str, Any]:
    try:
        return _parse(path.read_text(encoding="utf-8"), str(path))
    except OSError as exc:
        raise TrustReplayError(f"Cannot read replay source {path}: {exc}") from exc

def _read_rows(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    yield _parse(line, f"{path}:{number}")
    except OSError as exc:
        raise TrustReplayError(f"Cannot read replay source {path}: {exc}") from exc

def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if _canonical(actual) != _canonical(expected):
        raise TrustReplayError(f"Replay source {label} mismatch: expected {expected!r}, got {actual!r}")

def _identity(row: dict[str, Any], label: str) -> tuple[int, int, int]:
    values = []
    for field in ("trajectory_idx", "episode_position", "attempt_idx"):
        value = row.get(field)
        if type(value) is not int or value < 1:
            raise TrustReplayError(f"Replay {label} has invalid {field}: {value!r}")
        values.append(value)
    trajectory, episode, attempt = values
    negotiation = f"t{trajectory:03d}-e{episode:03d}"
    _require_equal(row.get("negotiation_id"), negotiation, f"{label} negotiation_id")
    _require_equal(row.get("attempt_id"), f"{negotiation}-a{attempt:03d}", f"{label} attempt_id")
    for alias, value in (("run_idx", trajectory), ("episode_idx", episode)):
        if alias in row:
            _require_equal(row[alias], value, f"{label} {alias}")
    return trajectory, episode, attempt

def _validate_matrix(value: Any, agent_ids: set[str], label: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, dict) or set(value) != agent_ids:
        raise TrustReplayError(f"Replay {label} trust_before must contain exactly all observers")
    for observer, view in value.items():
        if not isinstance(view, dict) or set(view) != agent_ids - {observer}:
            raise TrustReplayError(f"Replay {label} trust_before[{observer}] must contain every counterpart and no self-score")
        for target, score in view.items():
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise TrustReplayError(f"Replay {label} invalid trust score for {observer} -> {target}: {score!r}")
    return copy.deepcopy(value)

def _matching_config(experiment: dict[str, Any], config: GameConfig) -> dict[str, Any]:
    donor = experiment.get("run_config")
    if not isinstance(donor, dict):
        raise TrustReplayError("Replay source experiment.json is missing run_config")
    for field, expected in (
        ("supplement_prompt_protocol", SUPPLEMENT_PROMPT_PROTOCOL),
        ("design_version", LONGITUDINAL_DESIGN_VERSION),
        ("seed_scheme_version", SEED_SCHEME_VERSION),
        ("threshold_action_policy", THRESHOLD_ACTION_POLICY),
        ("base_seed", config.game.base_seed),
        ("scenario_order_policy", config.game.scenario_order_policy),
        ("rounds_per_episode_multiplier", config.game.rounds_per_episode_multiplier),
        ("trust_mode", config.trust.mode),
        ("trust_is_frozen", False),
        ("trust_condition_enabled", True),
        ("trust_initial_score", config.trust.initial_score),
        ("trust_decay", config.trust.decay),
        ("trust_metric_weights", config.trust.metric_weights),
        ("trust_min_blocking_rejects", config.trust.min_blocking_rejects),
        ("trust_min_sample_rounds", config.trust.min_sample_rounds),
        ("malicious_config", config.malicious.model_dump()),
        ("configured_agent_models", config.agent_models),
    ):
        actual = donor.get(field, False) if field == "trust_is_frozen" else donor.get(field)
        _require_equal(actual, expected, f"run_config.{field}")
    if donor.get("trust_replay") or experiment.get("trust_replay"):
        raise TrustReplayError("Replay source must be an assessed S run, not another replay")
    if donor.get("trust_random") or donor.get("trust_random_each_episode"):
        raise TrustReplayError("Replay source must be an assessed S run, not random trust")
    if config.trust.mode == "none":
        raise TrustReplayError("Replay source cannot use trust.mode='none'")
    for field in ("trust_mode", "design_version"):
        _require_equal(experiment.get(field), donor[field], f"experiment.{field}")
    for field, required in (("num_trajectories", config.game.num_trajectories), ("num_episodes", config.game.num_episodes)):
        count = donor.get(field)
        if type(count) is not int or count < required:
            raise TrustReplayError(f"Replay source has too few {field}: need {required}, found {count!r}")
    source_prompts = donor.get("prompt_config")
    if not isinstance(source_prompts, dict):
        raise TrustReplayError("Replay source is missing prompt_config")
    _require_equal(source_prompts.get("trust_decision_hint_level"), "scores_only", "source guidance")
    prompts = config.prompts.model_dump()
    source_prompts = dict(source_prompts)
    prompts.pop("trust_decision_hint_level", None)
    source_prompts.pop("trust_decision_hint_level", None)
    _require_equal(source_prompts, prompts, "prompt_config (excluding guidance)")
    source_llm = donor.get("llm_config")
    if not isinstance(source_llm, dict):
        raise TrustReplayError("Replay source is missing llm_config")
    source_llm = dict(source_llm)
    source_llm.pop("dry_run", None)
    target_llm = config.llm.model_dump()
    target_llm.pop("dry_run", None)
    _require_equal(source_llm, target_llm, "llm_config (excluding dry_run)")

    return {
        "supplement_prompt_protocol": donor["supplement_prompt_protocol"],
        "design_version": donor["design_version"],
        "seed_scheme_version": donor["seed_scheme_version"],
        "threshold_action_policy": donor["threshold_action_policy"],
        "base_seed": donor["base_seed"],
        "scenario_order_policy": donor["scenario_order_policy"],
        "trust_mode": donor["trust_mode"],
        "configured_agent_models": donor["configured_agent_models"],
        "agent_models": experiment.get("agent_models"),
        "resolved_client_provenance": donor.get("resolved_client_provenance", {}),
        "prompt_config": donor["prompt_config"],
        "llm_config": source_llm,
        "malicious_config": donor["malicious_config"],
    }

class TrustReplaySchedule:
    def __init__(self, payload: dict[str, Any], digest: str) -> None:
        self._payload = copy.deepcopy(payload)
        self.schedule_sha256 = digest
        self.source_run = payload["source_run"]
        self._cells = {
            (cell["trajectory_idx"], cell["episode_position"]): cell
            for cell in self._payload["cells"]
        }

    @classmethod
    def load(
        cls,
        source_run: str | Path,
        *,
        config: GameConfig,
        expected_cells: dict[tuple[int, int], dict[str, Any]],
        output_dir: str | Path,
        resume: bool,
    ) -> "TrustReplaySchedule":
        source = Path(source_run).expanduser()
        source = (PROJECT_ROOT / source).resolve() if not source.is_absolute() else source.resolve()
        output = Path(output_dir).expanduser().resolve()
        if source == output or source in output.parents or output in source.parents:
            raise TrustReplayError("Replay source and output directories must not overlap")
        try:
            source_label = str(source.relative_to(PROJECT_ROOT))
        except ValueError:
            source_label = str(source)
        expected_keys = {
            (trajectory, episode)
            for trajectory in range(1, config.game.num_trajectories + 1)
            for episode in range(1, config.game.num_episodes + 1)
        }
        if set(expected_cells) != expected_keys:
            raise TrustReplayError("Replay expected_cells must cover exactly all configured trajectories and episodes")
        experiment = _read_object(source / "experiment.json")
        settings = _matching_config(experiment, config)
        completed: dict[tuple[int, int, int], dict[str, Any]] = {}
        for row in _read_rows(source / "incidents.jsonl"):
            if row.get("type") not in {"run_completed", "run_failure"}:
                continue
            identity = _identity(row, "incident")
            if identity[:2] not in expected_keys:
                continue
            if row["type"] == "run_failure":
                completed.pop(identity, None)
            else:
                old = completed.get(identity)
                if old is not None and old != row:
                    raise TrustReplayError(f"Conflicting completion incidents for {row['attempt_id']}")
                completed[identity] = row
        latest: dict[tuple[int, int], tuple[int, int, int]] = {}
        for identity in completed:
            if identity[:2] not in latest or identity[2] > latest[identity[:2]][2]:
                latest[identity[:2]] = identity
        missing = expected_keys - set(latest)
        if missing:
            raise TrustReplayError(f"Replay source is missing completed cells: {sorted(missing)[:10]}")
        selected: dict[tuple[int, int], dict[str, Any]] = {}
        for row in _read_rows(source / "episodes.jsonl"):
            identity = _identity(row, "episode")
            key = identity[:2]
            if key not in expected_keys or latest[key] != identity:
                continue
            if key in selected:
                if selected[key] != row:
                    raise TrustReplayError(f"Conflicting completed episode rows for {row['attempt_id']}")
            else:
                selected[key] = row
        missing = expected_keys - set(selected)
        if missing:
            raise TrustReplayError(f"Replay source is missing episode rows for completed cells: {sorted(missing)[:10]}")
        cells = []
        for key in sorted(expected_keys):
            row = selected[key]
            expected = expected_cells[key]
            label = row["negotiation_id"]
            for field, value in expected.items():
                actual = row.get(field, False) if field == "trust_is_frozen" else row.get(field)
                _require_equal(actual, value, f"{label}.{field}")
            for field, value in (
                ("supplement_prompt_protocol", SUPPLEMENT_PROMPT_PROTOCOL),
                ("trust_mode", config.trust.mode),
                ("trust_is_frozen", False),
                ("trust_condition_enabled", True),
                ("prompt_version", config.prompts.prompt_version),
                ("pass_rule", config.game.pass_rule),
                ("design_version", LONGITUDINAL_DESIGN_VERSION),
            ):
                actual = row.get(field, False) if field == "trust_is_frozen" else row.get(field)
                _require_equal(actual, value, f"{label}.{field}")
            if row.get("trust_replay"):
                raise TrustReplayError(f"Replay source {label} was itself replayed")
            if row.get("trust_random"):
                raise TrustReplayError(f"Replay source {label} used random trust")
            if "run_seed" in row:
                _require_equal(row["run_seed"], row.get("episode_seed"), f"{label}.run_seed")
            incident = completed[latest[key]]
            for field in ("trajectory_seed", "episode_seed", "scenario_idx", "scenario_hash"):
                _require_equal(incident.get(field), row.get(field), f"{label} completion {field}")
            agent_ids = set(expected["agent_types"])
            _require_equal(row.get("trust_visibility_by_agent"), dict.fromkeys(agent_ids, True), f"{label} trust visibility")
            matrix = _validate_matrix(row.get("trust_before"), agent_ids, label)
            cells.append({
                **copy.deepcopy(expected),
                "source_negotiation_id": label,
                "source_attempt_id": row["attempt_id"],
                "source_attempt_idx": row["attempt_idx"],
                "trust_before": matrix,
            })
        payload = {
            "schema_version": 1,
            "protocol": REPLAY_PROTOCOL,
            "source_run": source_label,
            "source_settings": settings,
            "num_trajectories": config.game.num_trajectories,
            "num_episodes": config.game.num_episodes,
            "cells": cells,
        }
        digest = _digest(payload)
        destination = output / SCHEDULE_FILENAME
        if destination.exists():
            saved = _read_object(destination)
            saved_digest = saved.pop("schedule_sha256", None)
            if saved_digest != _digest(saved):
                raise TrustReplayError("Pinned trust replay schedule checksum is invalid")
            if saved_digest != digest or saved != payload:
                raise TrustReplayError("Replay source or selection changed since trust_replay.json was pinned")
        else:
            if resume:
                raise TrustReplayError("Cannot resume without the pinned trust_replay.json schedule")
            artifacts = ("checkpoint.json", "experiment.json", "episodes.jsonl", "incidents.jsonl")
            if any((output / name).exists() and (output / name).stat().st_size for name in artifacts):
                raise TrustReplayError("Existing run is missing its pinned trust_replay.json; cannot attach a new schedule")
            output.mkdir(parents=True, exist_ok=True)
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output, prefix=".trust_replay-", suffix=".tmp", delete=False) as handle:
                    temporary = handle.name
                    json.dump({**payload, "schedule_sha256": digest}, handle, sort_keys=True, indent=2, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if temporary is not None and Path(temporary).exists():
                    Path(temporary).unlink()
        return cls(payload, digest)

    def snapshot(self, trajectory_idx: int, episode_position: int) -> dict[str, dict[str, float]]:
        try:
            return copy.deepcopy(self._cells[(trajectory_idx, episode_position)]["trust_before"])
        except KeyError as exc:
            raise TrustReplayError(f"No pinned trust for trajectory {trajectory_idx}, episode {episode_position}") from exc

    def validate_client_provenance(
        self, current_provenance: dict[str, dict[str, Any]], *, allow_stub: bool,
    ) -> None:
        donor = self._payload["source_settings"]["resolved_client_provenance"]
        if not isinstance(donor, dict) or not donor:
            raise TrustReplayError("Replay source is missing resolved client provenance")
        if not isinstance(current_provenance, dict) or set(donor) != set(current_provenance):
            raise TrustReplayError("Replay client provenance must cover exactly the donor's agents")
        semantic_fields = (
            "resolved_model", "api_version", "reasoning_prefixes", "thinking_budget",
            "supports_temperature", "supports_seed", "reasoning_effort",
            "is_reasoning_model", "merge_system_into_user", "supports_response_format",
            "openai_response_format_protocol", "structured_output_protocol",
            "provider", "backend", "max_token_parameter", "thinking_mode", "thinking_keep",
            "structured_output_reformat_attempts", "structured_output_regeneration_attempts",
            "structured_output_transport_retry_attempts", "scratchpad_limits",
        )
        for agent_id, source in donor.items():
            current = current_provenance[agent_id]
            if not isinstance(source, dict) or not isinstance(current, dict):
                raise TrustReplayError(f"Replay client provenance for {agent_id} must be a mapping")
            label = f"client {agent_id}"
            if not isinstance(source.get("configured_model"), str):
                raise TrustReplayError(f"Replay source {label} is missing configured_model")
            _require_equal(current.get("configured_model"), source["configured_model"], f"{label}.configured_model")
            if type(source.get("is_stub")) is not bool or type(current.get("is_stub")) is not bool:
                raise TrustReplayError(f"Replay {label} must identify its execution backend")
            if current["is_stub"] and allow_stub:
                continue
            _require_equal(current["is_stub"], source["is_stub"], f"{label}.is_stub")
            if not current["is_stub"]:
                _require_equal(current.get("client_class"), source.get("client_class"), f"{label}.client_class")
            for field in semantic_fields:
                if field in source:
                    if field not in current:
                        raise TrustReplayError(f"Replay {label} is missing recorded capability {field}")
                    _require_equal(current[field], source[field], f"{label}.{field}")

    def provenance(self, trajectory_idx: int, episode_position: int) -> dict[str, Any]:
        cell = self._cells[(trajectory_idx, episode_position)]
        return {
            "protocol": REPLAY_PROTOCOL,
            "source_run": self.source_run,
            "schedule_sha256": self.schedule_sha256,
            "source_negotiation_id": cell["source_negotiation_id"],
            "source_attempt_id": cell["source_attempt_id"],
            "source_trajectory_idx": trajectory_idx,
            "source_episode_position": episode_position,
            "source_field": "trust_before",
            "assessment_enabled": False,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_PROTOCOL,
            "source_run": self.source_run,
            "schedule_sha256": self.schedule_sha256,
            "source_mode": self._payload["source_settings"]["trust_mode"],
            "num_trajectories": self._payload["num_trajectories"],
            "num_episodes": self._payload["num_episodes"],
            "num_cells": len(self._cells),
            "source_field": "trust_before",
            "assessment_enabled": False,
        }
