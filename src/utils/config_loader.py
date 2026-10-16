from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.utils.types import (
    AGENT_TYPE_IDS,
    AgentConfig,
    AgentEndpointConfig,
    GameConfig,
    Issue,
    Party,
    ScenarioConfig,
)

class ConfigError(ValueError):
    pass

def _read_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"Expected YAML mapping at top level in {resolved}")
    return data

def load_model_registry(path: str | Path) -> dict[str, AgentEndpointConfig]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    data = _read_yaml(resolved)
    registry: dict[str, AgentEndpointConfig] = {}
    for model_name, raw in data.items():
        if not isinstance(raw, dict):
            raise ConfigError(
                f"model_registry entry '{model_name}' must be a mapping, got {type(raw).__name__}"
            )
        if not str(model_name).lower().startswith("gemini") and not raw.get("base_url"):
            raise ConfigError(
                f"model_registry entry '{model_name}' requires base_url; "
                "only native Gemini models may omit it"
            )
        try:
            registry[model_name] = AgentEndpointConfig.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid model_registry entry '{model_name}': {exc}"
            ) from exc
    return registry

def load_game_config(path: str | Path) -> GameConfig:
    try:
        raw = _read_yaml(path)
        return GameConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid game config {path}: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"Invalid game config {path}: {exc}") from exc

def load_scenario(path: str | Path):
    from src.game.scenario import Scenario

    raw = _read_yaml(path)
    try:
        parsed = ScenarioConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid scenario config {path}: {exc}") from exc

    project = parsed.project
    parties_raw = project.get("parties")
    if not isinstance(parties_raw, list) or not parties_raw:
        raise ConfigError("scenario project.parties must be a non-empty list")

    parties = [Party.model_validate(item) for item in parties_raw]
    party_names = {party.id: party.display_name for party in parties}
    party_ids = set(party_names)

    issue_by_id = {issue.id: issue for issue in parsed.issues}
    if len(issue_by_id) != len(parsed.issues):
        raise ConfigError("scenario issues must have unique ids")
    legal_options = {
        issue.id: {option.id for option in issue.options}
        for issue in parsed.issues
    }

    agents: dict[str, AgentConfig] = {}
    for raw_agent in parsed.agents:
        agent_id = raw_agent.get("id")
        if agent_id not in party_ids:
            raise ConfigError(f"agent '{agent_id}' does not match any declared party")
        if agent_id in agents:
            raise ConfigError(f"duplicate agent id '{agent_id}' in scenario")

        issue_max_scores = raw_agent.get("issue_max_scores")
        scores = raw_agent.get("scores")
        min_threshold = raw_agent.get("min_threshold")
        if not isinstance(issue_max_scores, dict):
            raise ConfigError(f"agent '{agent_id}' missing issue_max_scores mapping")
        if not isinstance(scores, dict):
            raise ConfigError(f"agent '{agent_id}' missing scores mapping")
        if not isinstance(min_threshold, int):
            raise ConfigError(f"agent '{agent_id}' min_threshold must be an integer")
        if not 0 <= min_threshold <= 100:
            raise ConfigError(
                f"agent '{agent_id}' min_threshold must be in [0, 100], got {min_threshold}"
            )

        max_sum = 0
        if set(issue_max_scores) != set(issue_by_id):
            raise ConfigError(
                f"agent '{agent_id}' issue_max_scores must cover exactly the issues "
                f"{sorted(issue_by_id)}"
            )
        if set(scores) != set(issue_by_id):
            raise ConfigError(
                f"agent '{agent_id}' scores must cover exactly the issues {sorted(issue_by_id)}"
            )

        for issue_id, issue in issue_by_id.items():
            legal_option_ids = legal_options[issue_id]
            issue_scores = scores[issue_id]
            if set(issue_scores) != legal_option_ids:
                raise ConfigError(
                    f"agent '{agent_id}' scores for issue '{issue_id}' must contain exactly "
                    f"the legal options {sorted(legal_option_ids)}"
                )
            if any(not isinstance(value, int) for value in issue_scores.values()):
                raise ConfigError(
                    f"agent '{agent_id}' scores for issue '{issue_id}' must all be integers"
                )
            expected_max = issue_max_scores[issue_id]
            actual_max = max(issue_scores.values())
            if actual_max != expected_max:
                raise ConfigError(
                    f"agent '{agent_id}' issue '{issue_id}' has max score {actual_max}, "
                    f"expected {expected_max}"
                )
            max_sum += expected_max

        if max_sum != 100:
            raise ConfigError(
                f"agent '{agent_id}' issue_max_scores must sum to 100, got {max_sum}"
            )

        agent_type = raw_agent.get("agent_type", "")
        if agent_type and agent_type not in AGENT_TYPE_IDS:
            raise ConfigError(
                f"agent '{agent_id}' has unknown agent_type '{agent_type}'; "
                f"must be one of {sorted(AGENT_TYPE_IDS)} or omitted"
            )

        enriched_agent = {
            **raw_agent,
            "display_name": party_names[agent_id],
            "is_malicious": False,
        }
        try:
            agents[agent_id] = AgentConfig.model_validate(enriched_agent)
        except ValidationError as exc:
            raise ConfigError(
                f"Invalid agent configuration for '{agent_id}' in {path}: {exc}"
            ) from exc

    if set(agents) != party_ids:
        missing = sorted(party_ids - set(agents))
        raise ConfigError(f"scenario missing agent configs for parties: {missing}")

    project_name = project.get("name")
    shared_description = project.get("shared_description")
    if not isinstance(project_name, str) or not project_name.strip():
        raise ConfigError("scenario project.name must be a non-empty string")
    if not isinstance(shared_description, str) or not shared_description.strip():
        raise ConfigError("scenario project.shared_description must be a non-empty string")

    return Scenario(
        project_name=project_name,
        shared_description=shared_description,
        parties=parties,
        issues=parsed.issues,
        agent_configs=agents,
        _source_path=str(Path(path)),
        difficulty=parsed.difficulty,
    )
