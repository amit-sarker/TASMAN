from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from src.BRS.config import BRSConfig

class SubOption(BaseModel):
    id: str
    description: str

class Issue(BaseModel):
    id: str
    name: str
    description: str
    options: list[SubOption]

class Party(BaseModel):
    id: str
    display_name: str

class Deal(BaseModel):
    choices: dict[str, str]

class Vote(BaseModel):
    agent_id: str
    accept: bool
    reason: str

    trust_in_proposer: Optional[float] = None
    vote_type: str = ""

class RoundRecord(BaseModel):
    round_idx: int
    proposer_id: str
    deal: Deal
    rationale: str
    votes: list[Vote]
    proposer_implicit_accept: bool = True
    passed: bool
    scratchpads: dict[str, str] = Field(default_factory=dict)
    public_answers: dict[str, str] = Field(default_factory=dict)

class EpisodeRecord(BaseModel):
    episode_idx: int

    run_idx: int = 1
    trajectory_idx: Optional[int] = None
    episode_position: Optional[int] = None
    trajectory_seed: int = 0
    episode_seed: Optional[int] = None
    scenario_idx: int = 0
    design_version: str = "independent_longitudinal_trajectories_v2"
    negotiation_id: str = ""
    attempt_idx: int = 1
    attempt_id: str = ""
    scenario_path: str = ""
    difficulty: str = ""
    malicious_agent_id: str

    agent_types: dict[str, str] = Field(default_factory=dict)
    rounds: list[RoundRecord]
    success: bool
    final_deal: Optional[Deal]
    utilities: dict[str, int]
    trust_before: dict[str, dict[str, float]]
    trust_after: dict[str, dict[str, float]] = Field(default_factory=dict)

    trust_replay: dict[str, Any] = Field(default_factory=dict)
    trust_random: dict[str, Any] = Field(default_factory=dict)
    brs_state_before: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    brs_state_after: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    brs_feedback_key: str = ""
    rounds_to_deal: Optional[int] = None

    trust_mode: str = ""
    trust_is_frozen: bool = False

    trust_condition_enabled: bool = False
    trust_visible: bool = False
    trust_visibility_by_agent: dict[str, bool] = Field(default_factory=dict)
    prompt_version: str = ""
    trust_visibility_policy: str = ""
    thresholds_by_agent: dict[str, int] = Field(default_factory=dict)
    run_seed: int = 0
    proposer_order: list[str] = Field(default_factory=list)
    scenario_hash: str = ""
    max_rounds: int = 0
    n_agents: int = 0

    @model_validator(mode="after")
    def synchronize_longitudinal_aliases(self) -> "EpisodeRecord":
        fields_set = self.model_fields_set

        if self.trajectory_idx is None:
            self.trajectory_idx = self.run_idx
        elif "run_idx" in fields_set and self.run_idx != self.trajectory_idx:
            raise ValueError("run_idx must equal trajectory_idx")
        else:
            self.run_idx = self.trajectory_idx

        if self.episode_position is None:
            self.episode_position = self.episode_idx
        elif self.episode_idx != self.episode_position:
            raise ValueError("episode_idx must equal episode_position")

        if self.episode_seed is None:
            self.episode_seed = self.run_seed
        elif "run_seed" in fields_set and self.run_seed != self.episode_seed:
            raise ValueError("run_seed must equal episode_seed")
        else:
            self.run_seed = self.episode_seed

        if not self.negotiation_id:
            self.negotiation_id = (
                f"t{self.trajectory_idx:03d}-e{self.episode_position:03d}"
            )
        if self.attempt_idx < 1:
            raise ValueError("attempt_idx must be at least 1")
        if not self.attempt_id:
            self.attempt_id = f"{self.negotiation_id}-a{self.attempt_idx:03d}"
        return self

AGENT_TYPE_IDS = (
    "malicious_blocker",
    "malicious_strategic",
    "hard_bargainer",
    "priority_purist",
    "general",
)

class AgentConfig(BaseModel):
    id: str
    display_name: str
    goal_description: str
    priority_narrative: str
    issue_max_scores: dict[str, int]
    scores: dict[str, dict[str, int]]
    min_threshold: int
    is_malicious: bool = False
    preference_examples_about_others: str
    model: Optional[str] = None

    agent_type: str = ""

    priority_description: str = ""

    @model_validator(mode="after")
    def normalize_malicious_type(self) -> "AgentConfig":
        if self.is_malicious or self.agent_type == "malicious_blocker":
            self.agent_type = "malicious_strategic"
        return self

@dataclass
class TrustScores:
    scores: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def initialize(cls, agent_ids: list[str], initial_score: float) -> "TrustScores":
        matrix: dict[str, dict[str, float]] = {}
        for focal_id in agent_ids:
            matrix[focal_id] = {
                other_id: initial_score
                for other_id in agent_ids
                if other_id != focal_id
            }
        return cls(scores=matrix)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, dict[str, float]]) -> "TrustScores":
        return cls(scores={focal: dict(others) for focal, others in snapshot.items()})

    def get(self, focal: str, other: str) -> float:
        return self.scores[focal][other]

    def set(self, focal: str, other: str, value: float) -> None:
        self.scores.setdefault(focal, {})[other] = value

    def view_for(self, focal: str) -> dict[str, float]:
        return dict(self.scores.get(focal, {}))

    def set_view_for(self, focal: str, view: dict[str, float]) -> None:
        self.scores[focal] = dict(view)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            focal_id: dict(other_scores)
            for focal_id, other_scores in self.scores.items()
        }

class ProposerOutput(BaseModel):
    scratchpad: str
    public_answer: str
    deal: Deal

class VoterOutput(BaseModel):
    scratchpad: str
    public_answer: str
    vote: bool
    reason: str

class TrustOutput(BaseModel):
    scratchpad: str

    scratchpad_by_agent: dict[str, str] = Field(default_factory=dict)
    trust_scores: dict[str, float]

    episode_signals: dict[str, float] = Field(default_factory=dict)
    metrics_weights: dict[str, float] = Field(default_factory=dict)

class OpeningRoundConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["model_generated", "self_optimal", "fixed", "centroid"]
    fixed_deal: Optional[str] = None

    @model_validator(mode="after")
    def validate_fixed_deal(self) -> "OpeningRoundConfig":
        if self.mode == "fixed" and not self.fixed_deal:
            raise ValueError("prompts.opening_round.fixed_deal must be set when mode='fixed'")
        return self

class PromptsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history_window_size: Optional[int] = None
    opening_round: OpeningRoundConfig

    trust_decision_hint_level: Literal["scores_only", "prescriptive"] = "prescriptive"

    @property
    def prompt_version(self) -> str:
        return "v5"

    @property
    def trust_guidance(self) -> str:
        return "prescriptive"

    @model_validator(mode="after")
    def validate_history_window_size(self) -> "PromptsConfig":
        if self.history_window_size is not None and self.history_window_size <= 0:
            raise ValueError("prompts.history_window_size must be positive or null")
        return self

class GameSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_episodes: int
    pass_rule: str
    rounds_per_episode_multiplier: int
    base_seed: int

    num_trajectories: Optional[int] = None
    num_runs: Optional[int] = None
    parallel_workers: int = 1
    scenario_order_policy: Literal["fixed", "cyclic_counterbalanced"] = "fixed"

    @model_validator(mode="after")
    def synchronize_trajectory_count(self) -> "GameSettings":
        supplied = self.model_fields_set
        if (
            "num_trajectories" in supplied
            and "num_runs" in supplied
            and self.num_trajectories != self.num_runs
        ):
            raise ValueError("game.num_runs must equal game.num_trajectories when both are set")

        count = self.num_trajectories if self.num_trajectories is not None else self.num_runs
        if count is None:
            count = 1
        if count < 1:
            raise ValueError("game.num_trajectories must be at least 1")
        if self.num_episodes < 1:
            raise ValueError("game.num_episodes must be at least 1")
        if self.rounds_per_episode_multiplier < 1:
            raise ValueError("game.rounds_per_episode_multiplier must be at least 1")
        if self.parallel_workers < 1:
            raise ValueError("game.parallel_workers must be at least 1")

        self.num_trajectories = count
        self.num_runs = count
        return self

class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    reasoning_model_prefixes: list[str] = []

    run_entropy: bool = True

    dry_run: bool = False

class TrustReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_run: str = Field(min_length=1, strict=True)

    @model_validator(mode="after")
    def validate_source_run(self) -> "TrustReplayConfig":
        if not self.source_run.strip():
            raise ValueError("trust.replay.source_run must be a nonempty run directory")
        return self

class TrustConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["metrics", "freeform", "none", "brs"] = "metrics"
    is_frozen: bool = False
    random_each_episode: bool = False
    initial_score: float
    decay: float = 0.7

    metric_weights: dict[str, float] = Field(default_factory=dict)
    min_blocking_rejects: int = 2
    min_sample_rounds: int = 3
    brs: Optional[BRSConfig] = None
    replay: Optional[TrustReplayConfig] = None

    @model_validator(mode="after")
    def validate_brs(self) -> "TrustConfig":
        if self.random_each_episode:
            if self.mode == "none":
                raise ValueError("trust.random_each_episode requires a trust-enabled mode")
            if self.is_frozen or self.replay is not None:
                raise ValueError(
                    "trust.random_each_episode cannot be combined with trust.is_frozen or trust.replay"
                )
        if self.replay is not None:
            if self.mode == "none":
                raise ValueError("trust.replay requires a trust-enabled mode")
            if self.is_frozen:
                raise ValueError("trust.replay and trust.is_frozen are mutually exclusive")
        if self.mode == "brs":
            if self.brs is None:
                self.brs = BRSConfig()
            if not self.is_frozen and not self.random_each_episode and self.initial_score != 0.5:
                raise ValueError("BRS uses a fixed Beta(1,1) prior: initial_score must be 0.5")
        return self

class MaliciousConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    agent_id: Optional[str]
    threshold: int

    @property
    def strategy(self) -> str:
        return "strategic"

class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    out_dir: str
    log_full_prompts: bool
    log_full_completions: bool
    console: bool = True
    console_private: bool = True

class AgentEndpointConfig(BaseModel):
    base_url: Optional[str] = None
    api_key_env: str
    api_version: Optional[str] = None
    thinking_budget: Optional[int] = None
    thinking_mode: Optional[
        Literal["provider_default", "enabled", "disabled"]
    ] = None
    thinking_keep: Optional[Literal["all"]] = None
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    ] = None

    is_reasoning_model: Optional[bool] = None
    merge_system_into_user: Optional[bool] = None
    supports_response_format: Optional[bool] = None
    openai_response_format_protocol: Optional[
        Literal["strict_fixed_json_schema_dynamic_maps_json_object_v1"]
    ] = None
    model_name: Optional[str] = None
    max_token_parameter: Literal[
        "max_completion_tokens",
        "max_tokens",
    ] = "max_completion_tokens"

    structured_output_protocol: Optional[
        Literal[
            "gemini_native_v1",
            "kimi_text_json_v1",
            "deepseek_text_json_v1",
            "openai_native_v1",
        ]
    ] = None
    structured_output_regeneration_attempts: int = Field(default=0, ge=0, le=1)
    structured_output_reformat_attempts: int = Field(default=0, ge=0, le=0)
    trajectory_cell_retry_attempts: int = Field(default=0, ge=0, le=10)
    scratchpad_limits: dict[str, StrictInt] = Field(default_factory=dict)
    scratchpad_recovery_targets: dict[str, list[StrictInt]] = Field(
        default_factory=dict
    )

    supports_temperature: bool = False

    supports_seed: bool = True

    require_seed_support: bool = False

    @model_validator(mode="after")
    def validate_structured_output_settings(self) -> "AgentEndpointConfig":
        if self.thinking_keep is not None and self.thinking_mode != "enabled":
            raise ValueError(
                "thinking_keep requires thinking_mode='enabled'"
            )
        if self.thinking_mode is not None and self.reasoning_effort is not None:
            raise ValueError(
                "thinking_mode and reasoning_effort are mutually exclusive"
            )
        if self.require_seed_support and not self.supports_seed:
            raise ValueError(
                "require_seed_support requires supports_seed=true"
            )
        if (
            self.structured_output_protocol is not None
            and self.openai_response_format_protocol is not None
            and self.structured_output_protocol != "openai_native_v1"
        ):
            raise ValueError(
                "structured_output_protocol and "
                "openai_response_format_protocol are mutually exclusive "
                "except for openai_native_v1"
            )
        allowed_limits = {
            "propose",
            "vote",
            "trust_per_target",
            "trust_total",
        }
        unexpected = sorted(set(self.scratchpad_limits) - allowed_limits)
        if unexpected:
            raise ValueError(
                "scratchpad_limits contains unsupported keys: "
                f"{unexpected}; expected only {sorted(allowed_limits)}"
            )
        non_positive = {
            key: value
            for key, value in self.scratchpad_limits.items()
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0
        }
        if non_positive:
            raise ValueError(
                "scratchpad_limits values must be positive integers: "
                f"{non_positive}"
            )
        unexpected_targets = sorted(
            set(self.scratchpad_recovery_targets) - allowed_limits
        )
        if unexpected_targets:
            raise ValueError(
                "scratchpad_recovery_targets contains unsupported keys: "
                f"{unexpected_targets}; expected only {sorted(allowed_limits)}"
            )
        if self.scratchpad_recovery_targets:
            raise ValueError(
                "scratchpad_recovery_targets is no longer used: valid actions "
                "with over-limit scratchpads are accepted and logged without "
                "another model call"
            )
        if self.structured_output_protocol is None:
            if (
                self.structured_output_reformat_attempts
                or self.structured_output_regeneration_attempts
                or self.trajectory_cell_retry_attempts
                or self.scratchpad_limits
                or self.scratchpad_recovery_targets
            ):
                raise ValueError(
                    "structured_output_protocol is required when configuring "
                    "structured-output retries, scratchpad limits, or recovery "
                    "targets"
                )
        if (
            self.trajectory_cell_retry_attempts
            and not self.structured_output_regeneration_attempts
        ):
            raise ValueError(
                "trajectory_cell_retry_attempts requires at least one "
                "structured_output_regeneration_attempt. Otherwise a seeded "
                "provider can repeat the same malformed decision on every "
                "cell attempt."
            )
        return self

class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game: GameSettings
    llm: LLMSettings
    trust: TrustConfig
    prompts: PromptsConfig
    malicious: MaliciousConfig
    logging: LoggingConfig

    agent_models: dict[str, str] = Field(default_factory=dict)

    scenario_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_pass_rule(self) -> "GameConfig":
        if self.game.pass_rule != "n_minus_1_total":
            raise ValueError("game.pass_rule must be 'n_minus_1_total'")
        if not 0.0 <= self.trust.initial_score <= 1.0:
            raise ValueError("trust.initial_score must be in [0, 1]")
        if not 0.0 <= self.trust.decay <= 1.0:
            raise ValueError("trust.decay must be in [0, 1]")
        if self.malicious.threshold <= 100:
            raise ValueError("malicious.threshold must be greater than 100")
        if not self.scenario_paths:
            raise ValueError("scenario_paths must be a non-empty list in game_config.yaml")
        return self

class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project: dict[str, Any]
    issues: list[Issue]
    agents: list[dict[str, Any]]
    difficulty: str = "medium"

    design: dict[str, Any] = Field(default_factory=dict)
