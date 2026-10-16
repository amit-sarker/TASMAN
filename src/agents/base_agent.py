from __future__ import annotations

from abc import ABC, abstractmethod
import json
import math
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from src.BRS.models import AgentFeedback

from src.agents.gemini_client import (
    GeminiSchemaValidationError,
    validate_json_schema_instance,
)
from src.agents.llm_client import ContentFilterError, LLMClient
from src.agents.structured_output import (
    BRSDeliveryError,
    GeminiStructuredOutputError,
    StructuredOutputError,
)
from src.agents.stub_client import StubLLMClient
from src.game.deal import (
    ProposerParseError,
    TrustParseError,
    VoterParseError,
    build_proposer_schema,
    build_trust_schema,
    build_voter_schema,
    format_deal,
    parse_deal_string,
    parse_proposer_output,
    parse_trust_output,
    parse_voter_output,
    redact_score_leaks,
)
from src.game.freeform_trust import (
    FreeformTrustParseError,
    build_freeform_trust_schema,
    parse_freeform_trust_output,
)
from src.game.scenario import Scenario
from src.agents.prompt_adapter import (
    build_proposer_user_message,
    build_system_prompt,
    build_voter_user_message,
)
from src.prompts.prompt_builder import (
    assert_no_private_leak,
    build_freeform_trust_update_user_message,
    build_trust_update_user_message,
    format_decision_history,
    format_history,
    format_trust_lines,
    format_trust_lines_compact,
)
from src.trust.calculation_audit import (
    audit_freeform_ema,
    audit_metrics_update,
)
from src.utils.seeding import (
    LONGITUDINAL_DESIGN_VERSION,
    derive_action_regeneration_seed,
    derive_call_seed,
    derive_episode_seed,
    derive_trajectory_seed,
)
from src.utils.types import AgentConfig, Deal, RoundRecord

ParsedOutputT = TypeVar("ParsedOutputT")

def _escape_literal_json_control_characters(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    for character in text:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            continue

        if escaped:
            output.append(character)
            escaped = False
            continue
        if character == "\\":
            output.append(character)
            escaped = True
            continue
        if character == '"':
            output.append(character)
            in_string = False
            continue
        if ord(character) < 0x20:
            output.append(json.dumps(character)[1:-1])
            continue
        output.append(character)
    return "".join(output)

def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GeminiSchemaValidationError(
                f"Duplicate JSON object key {key!r} is ambiguous"
            )
        result[key] = value
    return result

class BaseAgent(ABC):
    def __init__(
        self,
        *,
        config: AgentConfig,
        scenario: Scenario,
        llm_client: LLMClient | StubLLMClient,
        use_trust_scores: bool,
        trust_scores: dict[str, float],
        game_config: Any,
        logger: Any,
        model_name: str = "",
    ) -> None:
        self.agent_id = config.id
        self.config = config
        self.scenario = scenario
        self.llm_client = llm_client
        self.use_trust_scores = use_trust_scores
        self.trust_scores = dict(trust_scores)
        self.game_config = game_config
        self.logger = logger
        self.model_name = model_name
        self.privacy_violations = 0
        self.episode_idx: int = 0
        self.episode_position: int = 0
        self.trajectory_idx: int = 1
        self.run_idx: int = 1
        self.trajectory_seed: int = derive_trajectory_seed(
            getattr(self.game_config.game, "base_seed", 0),
            self.trajectory_idx,
        )
        self.episode_seed: int | None = None
        self.scenario_idx: int = 0
        self.scenario_path: str = ""
        self.scenario_hash: str = ""
        self.negotiation_id: str = ""
        self.attempt_idx: int = 1
        self.attempt_id: str = ""

    def _logger_identity(self) -> dict[str, Any]:
        return {
            "trajectory_idx": self.trajectory_idx,
            "trajectory_seed": self.trajectory_seed,
            "episode_position": self.episode_position or self.episode_idx,
            "scenario_idx": self.scenario_idx,
            "scenario_path": self.scenario_path,
            "scenario_hash": self.scenario_hash,
            "episode_seed": self.episode_seed or 0,
            "negotiation_id": self.negotiation_id,
            "attempt_idx": self.attempt_idx,
            "attempt_id": self.attempt_id,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
        }

    def compute_utility(self, deal: Deal) -> int:
        total = 0
        for issue in self.scenario.issues:
            if issue.id not in deal.choices:
                raise ValueError(f"Deal is missing issue '{issue.id}'")
            option_id = deal.choices[issue.id]
            if option_id not in self.config.scores[issue.id]:
                raise ValueError(
                    f"Illegal option '{option_id}' for issue '{issue.id}' in deal"
                )
            total += self.config.scores[issue.id][option_id]
        return max(0, min(100, total))

    def would_accept(self, deal: Deal) -> bool:
        return self.compute_utility(deal) >= self.config.min_threshold

    def decision_uses_trust(self) -> bool:
        return bool(
            self.use_trust_scores
            and getattr(self.game_config.trust, "mode", "none") != "none"
        )

    def _system_prompt(
        self,
        total_rounds: int,
        display_names: dict[str, str] | None = None,
        *,
        expose_trust_scores: bool | None = None,
    ) -> str:
        show_trust = self.decision_uses_trust() and expose_trust_scores is not False
        trust_lines = ""
        if show_trust and display_names is not None:
            trust_lines = self._trust_lines(display_names)
        return build_system_prompt(
            agent_config=self.config,
            scenario=self.scenario,
            total_rounds=total_rounds,
            n_agents=len(self.scenario.all_agent_ids()),
            trust_config=self.game_config.trust,
            use_trust_scores=show_trust,
            trust_lines=trust_lines,
            trust_decision_hint_level=getattr(
                self.game_config.prompts,
                "trust_decision_hint_level",
                "prescriptive",
            ),
        )

    def _trust_lines(self, display_names: dict[str, str]) -> str:
        return format_trust_lines(self.trust_scores, display_names)

    def _round_history_window_size(self) -> int:
        configured = self.game_config.prompts.history_window_size
        return configured or len(self.scenario.all_agent_ids())

    def _decision_history_text(
        self,
        history: list[RoundRecord],
        window_size: int,
    ) -> str:
        return format_decision_history(
            history,
            window_size,
            self.scenario.parties,
            [issue.id for issue in self.scenario.issues],
        )

    def _call_seed(self, round_idx: int, role: str) -> int:
        episode_seed = self.episode_seed
        if episode_seed is None:
            trajectory_seed = self.trajectory_seed or derive_trajectory_seed(
                getattr(self.game_config.game, "base_seed", 0),
                self.trajectory_idx,
            )
            episode_seed = derive_episode_seed(
                trajectory_seed,
                max(1, self.episode_position or self.episode_idx),
                f"scenario:{self.scenario_idx}",
            )
        return derive_call_seed(
            episode_seed,
            self.agent_id,
            round_idx,
            role,
        )

    def _uses_gemini_native_structured_output(self) -> bool:
        return bool(
            getattr(
                self.llm_client,
                "native_structured_output_enabled",
                False,
            )
        )

    def _consume_call_metadata(self) -> dict[str, Any]:
        consume = getattr(self.llm_client, "consume_call_metadata", None)
        return dict(consume() or {}) if callable(consume) else {}

    def _log_gemini_prompt_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        completion: str,
        role: str,
        round_idx: int,
        retry: bool,
        metadata: dict[str, Any],
    ) -> None:
        if not hasattr(self.logger, "log_prompt_completion"):
            return
        raw_completion_value = metadata.get("raw_completion")
        self.logger.log_prompt_completion(
            episode_idx=self.episode_idx,
            run_idx=self.run_idx,
            agent_id=self.agent_id,
            round_idx=round_idx,
            role=role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion=completion,

            raw_completion=(
                str(raw_completion_value)
                if raw_completion_value is not None
                else None
            ),
            retry=retry,
            recovery_stage=metadata.get("recovery_stage", "initial"),
            recovery_attempt=int(metadata.get("recovery_attempt", 0) or 0),
            provider=metadata.get("provider"),
            model=metadata.get("model"),
            finish_reason=metadata.get("finish_reason"),
            response_length=metadata.get("response_length"),
            returned_length=metadata.get("returned_length"),
            scratchpad_length=metadata.get("scratchpad_length"),
            scratchpad_max_target_length=metadata.get(
                "scratchpad_max_target_length"
            ),
            scratchpad_recovery_target=metadata.get(
                "scratchpad_recovery_target"
            ),
            scratchpad_recovery_total_target=metadata.get(
                "scratchpad_recovery_total_target"
            ),
            candidate_tokens=metadata.get("candidate_tokens"),
            thinking_tokens=metadata.get("thinking_tokens"),
            reasoning_content_length=metadata.get(
                "reasoning_content_length"
            ),
            reasoning_content_present=metadata.get(
                "reasoning_content_present"
            ),
            prompt_tokens=metadata.get("prompt_tokens"),
            request_seed=metadata.get("request_seed"),
            structured_output=metadata.get("structured_output"),
            mechanical_reformat=metadata.get("mechanical_reformat"),
            thinking_mode=metadata.get("thinking_mode"),
            max_token_parameter=metadata.get("max_token_parameter"),
            system_fingerprint=metadata.get("system_fingerprint"),
            **self._logger_identity(),
        )

    def _log_structured_output_event(
        self,
        *,
        role: str,
        round_idx: int,
        stage: str,
        attempt: int,
        success: bool,
        error: str | None,
        completion: str,
        metadata: dict[str, Any],
        terminal_outcome: bool = True,
    ) -> None:
        if not hasattr(self.logger, "log_structured_output_recovery"):
            return
        log_round_idx = round_idx + 1 if round_idx >= 0 else round_idx
        provider = str(
            metadata.get("provider")
            or getattr(self.llm_client, "provider", None)
            or "gemini"
        )
        self.logger.log_structured_output_recovery(
            agent_id=self.agent_id,
            episode_idx=self.episode_idx,
            run_idx=self.run_idx,
            round_idx=log_round_idx,
            role=role,
            provider=provider,
            model=self.model_name or getattr(self.llm_client, "model", ""),
            stage=stage,
            recovery_attempt=attempt,
            success=success,
            terminal_outcome=terminal_outcome,
            error=error,
            response_length=int(
                metadata.get("response_length", len(completion)) or 0
            ),
            returned_length=len(completion),
            scratchpad_length=metadata.get("scratchpad_length"),
            scratchpad_max_target_length=metadata.get(
                "scratchpad_max_target_length"
            ),
            scratchpad_limit=metadata.get("scratchpad_limit"),
            scratchpad_overage=metadata.get("scratchpad_overage"),
            scratchpad_per_target_limit=metadata.get(
                "scratchpad_per_target_limit"
            ),
            scratchpad_total_limit=metadata.get("scratchpad_total_limit"),
            scratchpad_total_overage=metadata.get(
                "scratchpad_total_overage"
            ),
            scratchpad_oversized_targets=metadata.get(
                "scratchpad_oversized_targets"
            ),
            scratchpad_recovery_target=metadata.get(
                "scratchpad_recovery_target"
            ),
            scratchpad_recovery_total_target=metadata.get(
                "scratchpad_recovery_total_target"
            ),
            finish_reason=metadata.get("finish_reason"),
            request_seed=metadata.get("request_seed"),
            candidate_tokens=metadata.get("candidate_tokens"),
            thinking_tokens=metadata.get("thinking_tokens"),
            **self._logger_identity(),
        )

    @staticmethod
    def _scratchpad_metadata_from_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        scratchpad = payload.get("scratchpad")
        if isinstance(scratchpad, str):
            return {
                "scratchpad_length": len(scratchpad),
                "scratchpad_max_target_length": None,
            }
        if isinstance(scratchpad, dict):
            lengths = {
                str(key): len(str(value)) for key, value in scratchpad.items()
            }
            flattened = "\n".join(
                f"{key}: {scratchpad[key]}" for key in scratchpad
            )
            return {
                "scratchpad_length": len(flattened),
                "scratchpad_max_target_length": max(lengths.values(), default=0),
                "scratchpad_target_lengths": lengths,
            }
        return {}

    def _scratchpad_limit_check(
        self,
        payload: dict[str, Any],
        role: str,
    ) -> tuple[str | None, dict[str, Any]]:
        limits = dict(getattr(self.llm_client, "scratchpad_limits", {}) or {})
        scratchpad_value = payload.get("scratchpad")
        if role in {"propose", "vote"}:
            limit = limits.get(role)
            scratchpad = (
                scratchpad_value if isinstance(scratchpad_value, str) else ""
            )
            details: dict[str, Any] = {}
            if limit:
                details = {
                    "scratchpad_limit": int(limit),
                    "scratchpad_overage": max(0, len(scratchpad) - int(limit)),
                }
            if limit and len(scratchpad) > limit:
                return (
                    f"{role} scratchpad has {len(scratchpad)} characters; "
                    f"maximum is {limit}",
                    details,
                )
            return None, details

        if role != "trust":
            return None, {}
        per_target_limit = limits.get("trust_per_target")
        scratchpad_by_agent = (
            dict(scratchpad_value) if isinstance(scratchpad_value, dict) else {}
        )
        oversized: dict[str, int] = {}
        if per_target_limit:
            oversized = {
                agent_id: len(str(reasoning))
                for agent_id, reasoning in scratchpad_by_agent.items()
                if len(str(reasoning)) > per_target_limit
            }
        total_limit = limits.get("trust_total")
        if isinstance(scratchpad_value, dict):
            complete_scratchpad = "\n".join(
                f"{key}: {scratchpad_value[key]}" for key in scratchpad_value
            )
        else:
            complete_scratchpad = str(scratchpad_value or "")
        target_overage = max(
            (
                length - int(per_target_limit)
                for length in oversized.values()
            ),
            default=0,
        )
        total_overage = (
            max(0, len(complete_scratchpad) - int(total_limit))
            if total_limit
            else 0
        )
        details = {
            "scratchpad_per_target_limit": per_target_limit,
            "scratchpad_total_limit": total_limit,
            "scratchpad_oversized_targets": oversized,
            "scratchpad_total_overage": total_overage,
            "scratchpad_overage": max(target_overage, total_overage),
        }
        if oversized:
            return (
                "trust scratchpad entries exceed the per-target maximum "
                f"of {per_target_limit}: {oversized}",
                details,
            )
        if total_limit and len(complete_scratchpad) > total_limit:
            return (
                f"trust scratchpad has {len(complete_scratchpad)} characters; "
                f"maximum is {total_limit}",
                details,
            )
        return None, details

    @staticmethod
    def _validate_gemini_trust_numeric_contract(
        payload: dict[str, Any],
    ) -> None:
        bounded_fields = (
            "trust_scores",
            "episode_signals",
            "metrics_weights",
        )
        for field in bounded_fields:
            values = payload.get(field)
            if values is None:
                continue
            if not isinstance(values, dict):
                raise GeminiSchemaValidationError(
                    f"Gemini trust field {field!r} must be an object"
                )
            for key, raw_value in values.items():
                if (
                    isinstance(raw_value, bool)
                    or not isinstance(raw_value, (int, float))
                    or not math.isfinite(float(raw_value))
                    or not 0.0 <= float(raw_value) <= 1.0
                ):
                    raise GeminiSchemaValidationError(
                        "Structured trust value "
                        f"{field}.{key} must be a finite number in [0, 1]; "
                        f"received {raw_value!r}"
                    )

        if "episode_signals" in payload and not payload.get("metrics_weights"):
            raise GeminiSchemaValidationError(
                "Structured free-form trust output must report at least one "
                "metrics_weights criterion"
            )

    @staticmethod
    def _parse_gemini_payload(
        completion: str,
        *,
        parser: Callable[[str], ParsedOutputT],
        schema: dict | None,
        role: str,
    ) -> tuple[ParsedOutputT, dict[str, Any], str, bool]:
        def parse_once(text: str) -> tuple[ParsedOutputT, dict[str, Any]]:
            payload = json.loads(
                text.strip(),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            if not isinstance(payload, dict):
                raise GeminiSchemaValidationError(
                    "Structured output must be a JSON object"
                )
            if schema is not None:
                validate_json_schema_instance(payload, schema)
            if role == "trust":
                BaseAgent._validate_gemini_trust_numeric_contract(payload)
            return parser(text), payload

        try:
            parsed, payload = parse_once(completion)
            return parsed, payload, completion, False
        except json.JSONDecodeError as first_error:
            repaired = _escape_literal_json_control_characters(completion)
            if repaired == completion:
                raise GeminiSchemaValidationError(
                    f"Response is not valid JSON: {first_error}"
                ) from first_error
            try:
                parsed, payload = parse_once(repaired)
            except json.JSONDecodeError as repair_error:
                raise GeminiSchemaValidationError(
                    f"Response is not valid JSON after control-character repair: "
                    f"{repair_error}"
                ) from repair_error
            return parsed, payload, repaired, True

    def _raise_gemini_structured_failure(
        self,
        *,
        role: str,
        round_idx: int,
        error: Exception | str,
        completion: str,
        metadata: dict[str, Any],
        log_parse_failure: bool = True,
    ) -> None:
        error_text = str(error)
        log_round_idx = round_idx + 1 if round_idx >= 0 else round_idx
        provider = str(
            metadata.get("provider")
            or getattr(self.llm_client, "provider", None)
            or "gemini"
        )
        self._log_structured_output_event(
            role=role,
            round_idx=round_idx,
            stage="unrecoverable",
            attempt=int(metadata.get("recovery_attempt", 0) or 0),
            success=False,
            error=error_text,
            completion=completion,
            metadata=metadata,
        )
        if log_parse_failure and hasattr(self.logger, "log_parse_failure"):
            self.logger.log_parse_failure(
                agent_id=self.agent_id,
                episode_idx=self.episode_idx,
                run_idx=self.run_idx,
                round_idx=log_round_idx,
                role=role,
                provider=provider,
                error=error_text,
                completion=completion,
                **self._logger_identity(),
            )
        error_class = getattr(
            self.llm_client,
            "structured_output_error_class",
            None,
        )
        if not isinstance(error_class, type) or not issubclass(
            error_class, StructuredOutputError
        ):
            error_class = GeminiStructuredOutputError
        raise error_class(
            f"{provider.capitalize()} structured output failed for "
            f"agent={self.agent_id!r}, "
            f"role={role!r}, round={log_round_idx}: {error_text}"
        )

    def _complete_gemini_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[str], ParsedOutputT],
        role: str,
        round_idx: int,
        schema: dict | None,
    ) -> ParsedOutputT:
        run_entropy = getattr(self.game_config.llm, "run_entropy", True)
        call_seed = self._call_seed(round_idx, role) if run_entropy else None
        original_user_prompt = user_prompt

        regeneration_attempts = min(
            1,
            max(
                0,
                int(
                    getattr(
                        self.llm_client,
                        "structured_output_regeneration_attempts",
                        0,
                    )
                ),
            ),
        )
        generation_errors: list[str] = []
        provider = str(
            getattr(self.llm_client, "provider", None) or "gemini"
        )

        for generation_attempt in range(regeneration_attempts + 1):
            generation_stage = (
                "initial" if generation_attempt == 0 else "regeneration"
            )
            generation_seed = call_seed
            if generation_attempt > 0 and call_seed is not None:
                generation_seed = derive_action_regeneration_seed(
                    call_seed,
                    self.attempt_idx,
                    generation_attempt,
                ) % (2**31)
            generation_user_prompt = original_user_prompt
            if run_entropy:
                generation_user_prompt = (
                    f"{generation_user_prompt}\n\n"
                    f"[session-nonce: {generation_seed}]"
                )

            assert_no_private_leak(
                f"{system_prompt}\n\n{generation_user_prompt}",
                self.agent_id,
                self.scenario.agent_configs,
            )
            try:
                completion = self.llm_client.complete(
                    system_prompt,
                    generation_user_prompt,
                    max_tokens=self.game_config.llm.max_tokens,
                    temperature=self.game_config.llm.temperature,
                    agent_id=self.agent_id,
                    round_idx=round_idx,
                    role=role,
                    schema=schema,
                    seed=generation_seed,
                    recovery_stage=generation_stage,
                    recovery_attempt=generation_attempt,
                    mechanical_reformat=False,
                )
            except ContentFilterError as exc:
                metadata = self._consume_call_metadata()
                metadata.setdefault("recovery_stage", generation_stage)
                metadata.setdefault("recovery_attempt", generation_attempt)
                metadata.setdefault("request_seed", generation_seed)
                raw_completion = str(metadata.get("raw_completion", ""))
                self._log_gemini_prompt_completion(
                    system_prompt=system_prompt,
                    user_prompt=generation_user_prompt,
                    completion=raw_completion,
                    role=role,
                    round_idx=round_idx,
                    retry=generation_attempt > 0,
                    metadata=metadata,
                )
                if generation_attempt == 0 and provider not in {
                    "kimi",
                    "deepseek",
                }:
                    raise
                self._log_structured_output_event(
                    role=role,
                    round_idx=round_idx,
                    stage=generation_stage,
                    attempt=generation_attempt,
                    success=False,
                    error=str(exc),
                    completion=raw_completion,
                    metadata=metadata,
                    terminal_outcome=False,
                )
                self._log_content_filter_incident(
                    round_idx,
                    role,
                    fallback_used=False,
                    recovery_stage=generation_stage,
                    recovery_attempt=generation_attempt,
                )
                if (
                    provider in {"kimi", "deepseek"}
                    and generation_attempt < regeneration_attempts
                ):
                    generation_errors.append(
                        f"{generation_stage} {generation_attempt}: {exc}"
                    )
                    continue
                self._raise_gemini_structured_failure(
                    role=role,
                    round_idx=round_idx,
                    error=f"regeneration {generation_attempt}: {exc}",
                    completion=raw_completion,
                    metadata=metadata,
                    log_parse_failure=False,
                )
                raise AssertionError("unreachable")
            except Exception as exc:
                if generation_attempt == 0:
                    raise
                metadata = self._consume_call_metadata()
                metadata.setdefault("provider", provider)
                metadata.setdefault(
                    "model",
                    self.model_name or getattr(self.llm_client, "model", ""),
                )
                metadata.setdefault("recovery_stage", "regeneration")
                metadata.setdefault("recovery_attempt", generation_attempt)
                metadata.setdefault("request_seed", generation_seed)
                metadata.setdefault("response_length", 0)
                metadata.setdefault("returned_length", 0)
                self._log_gemini_prompt_completion(
                    system_prompt=system_prompt,
                    user_prompt=generation_user_prompt,
                    completion="",
                    role=role,
                    round_idx=round_idx,
                    retry=True,
                    metadata=metadata,
                )
                self._log_structured_output_event(
                    role=role,
                    round_idx=round_idx,
                    stage="regeneration",
                    attempt=generation_attempt,
                    success=False,
                    error=str(exc),
                    completion="",
                    metadata=metadata,
                    terminal_outcome=False,
                )
                self._raise_gemini_structured_failure(
                    role=role,
                    round_idx=round_idx,
                    error=(
                        f"regeneration {generation_attempt} request failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    completion="",
                    metadata=metadata,
                    log_parse_failure=False,
                )
                raise AssertionError("unreachable")

            metadata = self._consume_call_metadata()
            metadata.setdefault("recovery_stage", generation_stage)
            metadata.setdefault("recovery_attempt", generation_attempt)
            metadata.setdefault("request_seed", generation_seed)
            generation_error: Exception | str | None = None
            control_repaired = False

            if metadata.get("incomplete"):
                generation_error = (
                    f"{provider.capitalize()} returned incomplete structured "
                    "output with "
                    "finish_reason="
                    f"{metadata.get('finish_reason')!r}"
                )
            else:
                try:
                    (
                        parsed,
                        baseline_payload,
                        accepted_completion,
                        control_repaired,
                    ) = self._parse_gemini_payload(
                        completion,
                        parser=parser,
                        schema=schema,
                        role=role,
                    )
                except (
                    ProposerParseError,
                    VoterParseError,
                    TrustParseError,
                    GeminiSchemaValidationError,
                    json.JSONDecodeError,
                ) as parse_error:
                    generation_error = parse_error

            if generation_error is None:
                break

            self._log_gemini_prompt_completion(
                system_prompt=system_prompt,
                user_prompt=generation_user_prompt,
                completion=completion,
                role=role,
                round_idx=round_idx,
                retry=generation_attempt > 0,
                metadata=metadata,
            )
            generation_errors.append(
                f"{generation_stage} {generation_attempt}: {generation_error}"
            )
            self._log_structured_output_event(
                role=role,
                round_idx=round_idx,
                stage=generation_stage,
                attempt=generation_attempt,
                success=False,
                error=str(generation_error),
                completion=completion,
                metadata=metadata,
                terminal_outcome=False,
            )
            repaired_candidate = _escape_literal_json_control_characters(
                completion
            )
            if repaired_candidate != completion:
                self._log_structured_output_event(
                    role=role,
                    round_idx=round_idx,
                    stage="control_character_repair",
                    attempt=generation_attempt,
                    success=False,
                    error=str(generation_error),
                    completion=repaired_candidate,
                    metadata=metadata,
                    terminal_outcome=False,
                )
            if generation_attempt >= regeneration_attempts:
                self._raise_gemini_structured_failure(
                    role=role,
                    round_idx=round_idx,
                    error=" | ".join(generation_errors),
                    completion=completion,
                    metadata=metadata,
                )
                raise AssertionError("unreachable")

        was_regenerated = generation_attempt > 0
        user_prompt = generation_user_prompt

        metadata.update(self._scratchpad_metadata_from_payload(baseline_payload))
        limit_error, limit_details = self._scratchpad_limit_check(
            baseline_payload,
            role,
        )
        metadata.update(limit_details)
        self._log_gemini_prompt_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion=completion,
            role=role,
            round_idx=round_idx,
            retry=was_regenerated,
            metadata=metadata,
        )

        if control_repaired:
            if was_regenerated:
                self._log_structured_output_event(
                    role=role,
                    round_idx=round_idx,
                    stage="control_character_repair",
                    attempt=generation_attempt,
                    success=True,
                    error=None,
                    completion=accepted_completion,
                    metadata=metadata,
                    terminal_outcome=False,
                )
            else:
                self._log_structured_output_event(
                    role=role,
                    round_idx=round_idx,
                    stage="initial",
                    attempt=0,
                    success=False,
                    error=(
                        "literal JSON control characters required "
                        "deterministic escaping"
                    ),
                    completion=completion,
                    metadata=metadata,
                    terminal_outcome=False,
                )

        if limit_error is None:
            self._log_structured_output_event(
                role=role,
                round_idx=round_idx,
                stage=(
                    "regeneration"
                    if was_regenerated
                    else "control_character_repair"
                    if control_repaired
                    else "initial"
                ),
                attempt=generation_attempt,
                success=True,
                error=None,
                completion=accepted_completion,
                metadata=metadata,
            )
            return parsed

        if was_regenerated:
            self._log_structured_output_event(
                role=role,
                round_idx=round_idx,
                stage="regeneration",
                attempt=generation_attempt,
                success=True,
                error=None,
                completion=accepted_completion,
                metadata=metadata,
                terminal_outcome=False,
            )
        elif control_repaired:
            self._log_structured_output_event(
                role=role,
                round_idx=round_idx,
                stage="control_character_repair",
                attempt=generation_attempt,
                success=True,
                error=None,
                completion=accepted_completion,
                metadata=metadata,
                terminal_outcome=False,
            )
        self._log_structured_output_event(
            role=role,
            round_idx=round_idx,
            stage="accepted_over_limit",
            attempt=generation_attempt,
            success=True,
            error=(
                f"{limit_error}; accepted without regeneration because the "
                "negotiation action is executable"
            ),
            completion=accepted_completion,
            metadata=metadata,
        )
        return parsed

    def _complete_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        parser: Callable[[str], ParsedOutputT],
        role: str,
        round_idx: int,
        schema: dict | None = None,
    ) -> ParsedOutputT:
        if self._uses_gemini_native_structured_output():
            return self._complete_gemini_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parser=parser,
                role=role,
                round_idx=round_idx,
                schema=schema,
            )

        run_entropy = getattr(self.game_config.llm, "run_entropy", True)
        call_seed = self._call_seed(round_idx, role) if run_entropy else None
        if run_entropy:
            user_prompt = f"{user_prompt}\n\n[session-nonce: {call_seed}]"

        assert_no_private_leak(
            f"{system_prompt}\n\n{user_prompt}",
            self.agent_id,
            self.scenario.agent_configs,
        )
        log_round_idx = round_idx + 1 if round_idx >= 0 else round_idx

        completion = self.llm_client.complete(
            system_prompt,
            user_prompt,
            max_tokens=self.game_config.llm.max_tokens,
            temperature=self.game_config.llm.temperature,
            agent_id=self.agent_id,
            round_idx=round_idx,
            role=role,
            schema=schema,
            seed=call_seed,
        )
        if hasattr(self.logger, "log_prompt_completion"):
            self.logger.log_prompt_completion(
                episode_idx=self.episode_idx,
                run_idx=self.run_idx,
                agent_id=self.agent_id,
                round_idx=round_idx,
                role=role,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                completion=completion,
                retry=False,
                **self._logger_identity(),
            )
        try:
            return parser(completion)
        except (ProposerParseError, VoterParseError, TrustParseError) as first_error:
            corrective_user = (
                f"{user_prompt}\n\n"
                "Your previous reply was not valid JSON or did not match the required schema. "
                "Retry now by responding with a single valid JSON object exactly as specified, "
                "with no text outside the JSON."
            )
            retry_completion = self.llm_client.complete(
                system_prompt,
                corrective_user,
                max_tokens=self.game_config.llm.max_tokens,
                temperature=self.game_config.llm.temperature,
                agent_id=self.agent_id,
                round_idx=round_idx,
                role=role,
                schema=schema,
                seed=call_seed,
            )
            if hasattr(self.logger, "log_prompt_completion"):
                self.logger.log_prompt_completion(
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    agent_id=self.agent_id,
                    round_idx=round_idx,
                    role=role,
                    system_prompt=system_prompt,
                    user_prompt=corrective_user,
                    completion=retry_completion,
                    retry=True,
                    **self._logger_identity(),
                )
            try:
                return parser(retry_completion)
            except (ProposerParseError, VoterParseError, TrustParseError) as second_error:
                if hasattr(self.logger, "log_parse_failure"):
                    self.logger.log_parse_failure(
                        agent_id=self.agent_id,
                        episode_idx=self.episode_idx,
                        run_idx=self.run_idx,
                        round_idx=log_round_idx,
                        role=role,
                        error=f"{first_error} | retry: {second_error}",
                        completion=retry_completion,
                        **self._logger_identity(),
                    )
                raise second_error

    def _redact_public_text(self, public_answer: str, round_idx: int, role: str) -> str:
        redacted, was_redacted = redact_score_leaks(
            public_answer,
            self.config.scores,
            self.config.min_threshold,
        )
        if was_redacted:
            self.privacy_violations += 1
            if hasattr(self.logger, "log_privacy_violation"):
                log_round_idx = round_idx + 1 if round_idx >= 0 else round_idx
                self.logger.log_privacy_violation(
                    agent_id=self.agent_id,
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    round_idx=log_round_idx,
                    role=role,
                    original_text=public_answer,
                    redacted_text=redacted,
                    **self._logger_identity(),
                )
        return redacted

    def _complete_proposer_turn(
        self,
        history: list[RoundRecord],
        round_idx: int,
        total_rounds: int,
        is_opening: bool,
        initial_deal: str | None,
        display_names: dict[str, str],
    ) -> tuple[Deal, str, str]:
        history_window_size = self._round_history_window_size()
        history_text = self._decision_history_text(
            history,
            history_window_size,
        )
        system_prompt = self._system_prompt(total_rounds, display_names)
        mode = self.game_config.prompts.opening_round.mode
        fixed_deal_hint = initial_deal if (is_opening and mode == "fixed") else None
        n_agents = len(self.scenario.all_agent_ids())
        user_prompt = build_proposer_user_message(
            round_idx=round_idx,
            total_rounds=total_rounds,
            history_text=history_text,
            use_trust_scores=self.decision_uses_trust(),
            initial_deal=fixed_deal_hint,
            needed_voter_accepts=n_agents - 2,
            is_malicious=self.config.is_malicious,
            num_voters=n_agents - 1,
            min_blocking_rejects=self.game_config.trust.min_blocking_rejects,
            trust_decision_hint_level=self.game_config.prompts.trust_decision_hint_level,
            trust_lines_compact=format_trust_lines_compact(
                self.trust_scores, display_names,
            ),
        )
        try:
            parsed = self._complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parser=lambda text: parse_proposer_output(text, self.scenario.issues),
                role="propose",
                round_idx=round_idx,
                schema=build_proposer_schema(self.scenario.issues),
            )
            public_answer = self._redact_public_text(parsed.public_answer, round_idx, "propose")
            return parsed.deal, public_answer, parsed.scratchpad
        except ContentFilterError:
            self._log_content_filter_incident(round_idx, "propose")
            fallback_deal = self._fallback_proposal(history, initial_deal)
            fallback_rationale = "I am restating a workable framework so the negotiation can continue."
            fallback_rationale = self._redact_public_text(fallback_rationale, round_idx, "propose")
            return fallback_deal, fallback_rationale, ""
        except ProposerParseError:
            fallback_deal = self._fallback_proposal(history, initial_deal)
            fallback_rationale = "I am restating a workable framework so the negotiation can continue."
            fallback_rationale = self._redact_public_text(fallback_rationale, round_idx, "propose")
            return fallback_deal, fallback_rationale, ""

    def _complete_voter_turn(
        self,
        deal: Deal,
        proposer_id: str,
        history: list[RoundRecord],
        round_idx: int,
        total_rounds: int,
        display_names: dict[str, str],
        proposer_rationale: str,
    ) -> tuple[bool, str, str]:
        history_window_size = self._round_history_window_size()
        history_text = self._decision_history_text(
            history,
            history_window_size,
        )
        system_prompt = self._system_prompt(total_rounds, display_names)
        user_prompt = build_voter_user_message(
            proposer_display_name=display_names[proposer_id],
            proposed_deal=format_deal(
                deal, [issue.id for issue in self.scenario.issues],
            ),
            proposer_rationale=proposer_rationale,
            round_idx=round_idx,
            total_rounds=total_rounds,
            history_text=history_text,
            use_trust_scores=self.decision_uses_trust(),
            trust_decision_hint_level=self.game_config.prompts.trust_decision_hint_level,
            is_malicious=self.config.is_malicious,
            proposer_trust=self.trust_scores.get(proposer_id),
        )
        try:
            parsed = self._complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parser=parse_voter_output,
                role="vote",
                round_idx=round_idx,
                schema=build_voter_schema(),
            )
            public_reason = self._redact_public_text(parsed.reason, round_idx, "vote")
            return parsed.vote, public_reason, parsed.scratchpad
        except ContentFilterError:
            self._log_content_filter_incident(round_idx, "vote")
            return False, "I cannot evaluate this proposal at this time.", ""
        except VoterParseError:
            return False, "I cannot support this proposal as stated.", ""

    def _log_content_filter_incident(
        self,
        round_idx: int,
        role: str,
        *,
        fallback_used: bool = True,
        recovery_stage: str | None = None,
        recovery_attempt: int | None = None,
    ) -> None:
        import sys
        disposition = (
            "using safe fallback"
            if fallback_used
            else "retrying the complete trajectory-episode cell"
        )
        print(
            f"  ⚠ Content filter triggered for agent '{self.agent_id}' "
            f"(role={role}, round={round_idx}) — {disposition}.",
            file=sys.stderr,
        )
        if hasattr(self.logger, "log_content_filter"):
            log_round_idx = round_idx + 1 if round_idx >= 0 else round_idx
            self.logger.log_content_filter(
                agent_id=self.agent_id,
                round_idx=log_round_idx,
                role=role,
                episode_idx=getattr(self, "episode_idx", None),
                run_idx=getattr(self, "run_idx", None),
                fallback_used=fallback_used,
                recovery_stage=recovery_stage,
                recovery_attempt=recovery_attempt,
                **self._logger_identity(),
            )

    def _fallback_proposal(self, history: list[RoundRecord], initial_deal: str | None) -> Deal:
        if history:
            return history[-1].deal
        if initial_deal:
            return parse_deal_string(initial_deal, self.scenario.issues)
        choices = {}
        for issue_id, score_map in self.config.scores.items():
            choices[issue_id] = max(score_map, key=score_map.get)
        return Deal(choices=choices)

    @staticmethod
    def _prompted_trust_scores(
        old_trust: dict[str, float],
        target_agent_ids: list[str],
    ) -> dict[str, float]:
        return {
            agent_id: float(f"{old_trust[agent_id]:.2f}")
            for agent_id in target_agent_ids
        }

    def _log_trust_ema_audit(
        self,
        *,
        episode_idx: int,
        target_agent_ids: list[str],
        payload: dict[str, Any] | None,
    ) -> None:
        if payload is None or not hasattr(
            self.logger, "log_trust_ema_calculation_inaccuracy"
        ):
            return
        row = {
            **payload,
            "checked_target_count": len(target_agent_ids),
            "inaccurate_target_count": len(
                payload.get("inaccurate_targets", {})
            ),
            "inaccurate_calculation_count": len(
                payload.get("inaccurate_targets", {})
            ),
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "is_malicious": bool(getattr(self.config, "is_malicious", False)),
            "agent_type": getattr(self.config, "agent_type", "") or "",
            "llm_output_preserved": True,
        }
        self.logger.log_trust_ema_calculation_inaccuracy(
            episode_idx=episode_idx,
            run_idx=self.run_idx,
            **row,
            **self._logger_identity(),
        )

    def _log_trust_metrics_audit(
        self,
        *,
        episode_idx: int,
        target_agent_ids: list[str],
        payload: dict[str, Any] | None,
    ) -> None:
        if payload is None or not hasattr(
            self.logger, "log_trust_metrics_calculation_inaccuracy"
        ):
            return
        target_issues = payload.get("target_issues", {})
        issues_by_target = {
            agent_id: list(details.get("issues", []))
            for agent_id, details in target_issues.items()
        }
        unverifiable_count = sum(
            any(
                issue.get("issue") == "unverifiable_calculation_trace"
                for issue in issues
            )
            for issues in issues_by_target.values()
        )
        inaccurate_target_count = sum(
            any(
                issue.get("issue") != "unverifiable_calculation_trace"
                for issue in issues
            )
            for issues in issues_by_target.values()
        )
        inaccurate_calculation_count = sum(
            issue.get("issue") != "unverifiable_calculation_trace"
            for issues in issues_by_target.values()
            for issue in issues
        )
        row = {
            **payload,
            "checked_target_count": len(target_agent_ids),
            "inaccurate_or_unverifiable_target_count": len(target_issues),
            "inaccurate_target_count": inaccurate_target_count,
            "inaccurate_calculation_count": inaccurate_calculation_count,
            "unverifiable_target_count": unverifiable_count,
            "agent_id": self.agent_id,
            "trust_mode": "metrics",
            "model_name": self.model_name,
            "is_malicious": bool(getattr(self.config, "is_malicious", False)),
            "agent_type": getattr(self.config, "agent_type", "") or "",
            "llm_output_preserved": True,
        }
        self.logger.log_trust_metrics_calculation_inaccuracy(
            episode_idx=episode_idx,
            run_idx=self.run_idx,
            **row,
            **self._logger_identity(),
        )

    def calculate_trust_after_episode(
        self,
        *,
        episode_idx: int,
        history: list[RoundRecord],
        total_rounds: int,
        display_names: dict[str, str],
        brs_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        trust_mode = getattr(self.game_config.trust, "mode", "metrics")

        if (getattr(self.game_config.trust, "replay", None) is not None
                or getattr(self.game_config.trust, "random_each_episode", False)):
            return dict(self.trust_scores)

        if getattr(self.game_config.trust, "is_frozen", False):
            self.trust_scores = {
                target: self.game_config.trust.initial_score
                for target in self.scenario.all_agent_ids() if target != self.agent_id
            }
            return dict(self.trust_scores)

        if trust_mode == "brs":
            return self._receive_brs_trust(
                episode_idx=episode_idx, display_names=display_names,
                scores=brs_scores,
            )

        if trust_mode == "none":
            return dict(self.trust_scores)

        if trust_mode == "freeform":
            return self._calculate_trust_freeform(
                episode_idx=episode_idx,
                history=history,
                total_rounds=total_rounds,
                display_names=display_names,
            )

        target_agent_ids = [
            agent_id
            for agent_id in self.scenario.all_agent_ids()
            if agent_id != self.agent_id
        ]

        history_text = format_history(history, len(history), self.scenario.parties)
        system_prompt = self._system_prompt(total_rounds, display_names)
        user_prompt = build_trust_update_user_message(
            history_text=history_text,
            trust_lines=self._trust_lines(display_names),
            target_agent_ids=target_agent_ids,
            trust_config=self.game_config.trust,
        )
        try:
            parsed = self._complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parser=lambda text: parse_trust_output(text, target_agent_ids),
                role="trust",
                round_idx=-1,
                schema=build_trust_schema(target_agent_ids),
            )
        except ContentFilterError:
            self._log_content_filter_incident(-1, "trust")
            if hasattr(self.logger, "log_trust_update_frozen"):
                self.logger.log_trust_update_frozen(
                    agent_id=self.agent_id,
                    episode_idx=episode_idx,
                    run_idx=self.run_idx,
                    trust_mode="metrics",
                    reason="content_filter",
                    frozen_trust=dict(self.trust_scores),
                    **self._logger_identity(),
                )
            return dict(self.trust_scores)
        except TrustParseError:
            if hasattr(self.logger, "log_trust_update_frozen"):
                self.logger.log_trust_update_frozen(
                    agent_id=self.agent_id,
                    episode_idx=episode_idx,
                    run_idx=self.run_idx,
                    trust_mode="metrics",
                    reason="parse_failure",
                    frozen_trust=dict(self.trust_scores),
                    **self._logger_identity(),
                )
            return dict(self.trust_scores)

        old_trust = dict(self.trust_scores)
        prompted_trust = self._prompted_trust_scores(
            old_trust, target_agent_ids
        )
        updated_trust = dict(parsed.trust_scores)

        self.trust_scores = updated_trust
        metrics_audit, ema_audit = audit_metrics_update(
            target_agent_ids=target_agent_ids,
            metrics_scratchpad=dict(parsed.scratchpad_by_agent)
            if parsed.scratchpad_by_agent
            else parsed.scratchpad,
            metrics_weights=dict(parsed.metrics_weights),
            trust_scores_prompted=prompted_trust,
            trust_scores_after=updated_trust,
            decay=self.game_config.trust.decay,
        )
        self._log_trust_metrics_audit(
            episode_idx=episode_idx,
            target_agent_ids=target_agent_ids,
            payload=metrics_audit,
        )
        self._log_trust_ema_audit(
            episode_idx=episode_idx,
            target_agent_ids=target_agent_ids,
            payload=ema_audit,
        )
        if hasattr(self.logger, "log_trust_update"):
            self.logger.log_trust_update(
                episode_idx=episode_idx,
                run_idx=self.run_idx,
                agent_id=self.agent_id,
                scratchpad=parsed.scratchpad,
                scratchpad_by_agent=dict(parsed.scratchpad_by_agent),
                trust_scores_before=old_trust,
                trust_scores_prompted=prompted_trust,
                trust_scores_after=dict(updated_trust),
                display_names=display_names,
                trust_mode=getattr(self.game_config.trust, "mode", "metrics"),
                model_name=self.model_name,
                is_malicious=bool(getattr(self.config, "is_malicious", False)),
                agent_type=getattr(self.config, "agent_type", "") or "",
                episode_signals={},
                metrics_weights=dict(parsed.metrics_weights),
                **self._logger_identity(),
            )
        return dict(updated_trust)

    def build_brs_feedback_request(
        self, history: list[RoundRecord], total_rounds: int,
        display_names: dict[str, str],
    ) -> dict[str, Any]:
        from src.BRS.brs import validate_agent_ids
        from src.BRS.feedback import build_agent_feedback_schema
        from src.prompts.feedback_prompt import build_feedback_user_message

        agent_ids = validate_agent_ids(self.scenario.all_agent_ids())
        if self.agent_id not in agent_ids or len(agent_ids) < 2:
            raise ValueError("BRS feedback observer must belong to a scenario with counterparts")
        targets = [target for target in agent_ids if target != self.agent_id]

        history_text = format_history(history, len(history), self.scenario.parties)
        seed = self._call_seed(-1, "trust_feedback") if getattr(
            self.game_config.llm, "run_entropy", True
        ) else None
        user_prompt = build_feedback_user_message(
            history_text, self.agent_id, targets, self._trust_lines(display_names),
        )
        if (seed is not None and getattr(self.llm_client, "structured_output_protocol", None)
                == "deepseek_text_json_v1"):
            user_prompt = f"{user_prompt}\n\n[session-nonce: {seed}]"
        return {
            "system": self._system_prompt(total_rounds, display_names),
            "user": user_prompt,
            "max_tokens": self.game_config.llm.max_tokens,
            "temperature": self.game_config.llm.temperature,
            "schema": build_agent_feedback_schema(targets),
            "seed": seed,
            "role": "trust_feedback",
        }

    def build_brs_feedback_correction_request(self, request: dict[str, Any]) -> dict[str, Any]:
        from src.BRS.response_policy import build_feedback_correction_request

        return build_feedback_correction_request(
            request, native_structured=self._uses_gemini_native_structured_output(),
            attempt_idx=getattr(self, "attempt_idx", 1),
        )

    def provide_brs_feedback(self, request: dict[str, Any]) -> AgentFeedback:
        from src.BRS.feedback import (
            FeedbackCallError, FeedbackOutputError, build_agent_feedback_schema,
            parse_agent_feedback_output_with_diagnostics,
        )
        from src.BRS.response_policy import RESPONSE_RECOVERY_POLICY

        targets = [target for target in self.scenario.all_agent_ids() if target != self.agent_id]
        if set(request) != {"system", "user", "max_tokens", "temperature", "schema", "seed", "role"}:
            raise ValueError("BRS feedback request has unexpected or missing fields")
        if request["schema"] != build_agent_feedback_schema(targets) or request["role"] != "trust_feedback":
            raise ValueError("BRS feedback request must assess exactly this agent's counterparts")
        self.brs_feedback_metadata = {}
        assessment = None
        repairs = []
        validation_error = None
        try:
            try:
                raw = self.llm_client.complete_feedback_once(**request)
            except Exception as exc:
                raise FeedbackCallError(f"Agent feedback request failed: {exc}") from exc
            try:
                assessment, repairs = parse_agent_feedback_output_with_diagnostics(raw, targets, allow_wrapping=True)
            except FeedbackOutputError as exc:
                validation_error = str(exc)
                raise
            return assessment
        finally:
            self.brs_feedback_metadata = dict(
                getattr(self.llm_client, "last_call_metadata", {}) or {}
            )
            self.brs_feedback_metadata["response_recovery_policy"] = RESPONSE_RECOVERY_POLICY
            if validation_error is not None:
                self.brs_feedback_metadata.update(
                    feedback_output_invalid=True, validation_error=validation_error,
                )
            if repairs:
                self.brs_feedback_metadata["local_parse_repairs"] = repairs
            if assessment is not None:
                empty_targets = [target for target, summary in assessment.scratchpad.items() if not summary]
                if empty_targets:
                    self.brs_feedback_metadata["empty_scratchpad_targets"] = empty_targets

    def _receive_brs_trust(
        self, *, episode_idx: int, display_names: dict[str, str],
        scores: dict[str, float] | None,
    ) -> dict[str, float]:
        from src.BRS.tool import TOOL_REQUEST_INSTRUCTION

        expected = set(self.scenario.all_agent_ids()) - {self.agent_id}
        if scores is None or set(scores) != expected or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1
            for value in scores.values()
        ):
            raise BRSDeliveryError("Staged BRS scores must cover exactly the other parties")
        authoritative = dict(scores)
        system = (
            "You are a participant in a completed negotiation. Retrieve your "
            "counterpart trust scores through the provided tool. After receiving "
            "the result, acknowledge receipt briefly without calculating new scores."
        )
        seed = self._call_seed(-1, "trust_tool") if getattr(
            self.game_config.llm, "run_entropy", True
        ) else None
        try:
            result = self.llm_client.trust_tool_roundtrip(
                system=system, user=TOOL_REQUEST_INSTRUCTION,
                scores=dict(authoritative), max_tokens=self.game_config.llm.max_tokens,
                temperature=self.game_config.llm.temperature, seed=seed,
            )
            if result != authoritative:
                raise BRSDeliveryError("Tool delivery changed authoritative scores")
        finally:
            self.brs_tool_metadata = dict(getattr(self.llm_client, "last_call_metadata", {}) or {})
        self.trust_scores = dict(authoritative)
        return dict(authoritative)

    def _calculate_trust_freeform(
        self,
        *,
        episode_idx: int,
        history: list[RoundRecord],
        total_rounds: int,
        display_names: dict[str, str],
    ) -> dict[str, float]:
        target_agent_ids = [
            agent_id
            for agent_id in self.scenario.all_agent_ids()
            if agent_id != self.agent_id
        ]
        history_text = format_history(history, len(history), self.scenario.parties)
        system_prompt = self._system_prompt(total_rounds, display_names)
        user_prompt = build_freeform_trust_update_user_message(
            history_text=history_text,
            trust_lines=self._trust_lines(display_names),
            target_agent_ids=target_agent_ids,
            trust_config=self.game_config.trust,
        )
        structured_scratchpad = True
        trust_schema = build_freeform_trust_schema(
            target_agent_ids,
            structured_scratchpad=structured_scratchpad,
        )
        uses_dynamic_json_object = getattr(
            self.llm_client,
            "uses_dynamic_map_json_object",
            lambda _schema: False,
        )
        strict_contract = bool(uses_dynamic_json_object(trust_schema))
        try:
            parsed = self._complete_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                parser=lambda text: parse_freeform_trust_output(
                    text,
                    target_agent_ids,
                    structured_scratchpad=structured_scratchpad,
                    strict_contract=strict_contract,
                ),
                role="trust",
                round_idx=-1,
                schema=trust_schema,
            )
        except ContentFilterError:
            self._log_content_filter_incident(-1, "trust")
            if hasattr(self.logger, "log_trust_update_frozen"):
                self.logger.log_trust_update_frozen(
                    agent_id=self.agent_id,
                    episode_idx=episode_idx,
                    run_idx=self.run_idx,
                    trust_mode="freeform",
                    reason="content_filter",
                    frozen_trust=dict(self.trust_scores),
                    **self._logger_identity(),
                )
            return dict(self.trust_scores)
        except FreeformTrustParseError:
            if hasattr(self.logger, "log_trust_update_frozen"):
                self.logger.log_trust_update_frozen(
                    agent_id=self.agent_id,
                    episode_idx=episode_idx,
                    run_idx=self.run_idx,
                    trust_mode="freeform",
                    reason="parse_failure",
                    frozen_trust=dict(self.trust_scores),
                    **self._logger_identity(),
                )
            return dict(self.trust_scores)

        old_trust = dict(self.trust_scores)
        prompted_trust = self._prompted_trust_scores(
            old_trust, target_agent_ids
        )
        updated_trust = dict(parsed.trust_scores)

        self.trust_scores = updated_trust
        ema_audit = audit_freeform_ema(
            trust_scores_prompted=prompted_trust,
            trust_scores_after=updated_trust,
            episode_signals=dict(parsed.episode_signals),
            decay=self.game_config.trust.decay,
            target_agent_ids=target_agent_ids,
        )
        self._log_trust_ema_audit(
            episode_idx=episode_idx,
            target_agent_ids=target_agent_ids,
            payload=ema_audit,
        )
        if hasattr(self.logger, "log_trust_update"):
            self.logger.log_trust_update(
                episode_idx=episode_idx,
                run_idx=self.run_idx,
                agent_id=self.agent_id,
                scratchpad=parsed.scratchpad,
                scratchpad_by_agent=dict(parsed.scratchpad_by_agent),
                trust_scores_before=old_trust,
                trust_scores_prompted=prompted_trust,
                trust_scores_after=dict(updated_trust),
                display_names=display_names,
                trust_mode="freeform",
                model_name=self.model_name,
                is_malicious=bool(getattr(self.config, "is_malicious", False)),
                agent_type=getattr(self.config, "agent_type", "") or "",
                episode_signals=dict(parsed.episode_signals),
                metrics_weights=dict(parsed.metrics_weights),
                **self._logger_identity(),
            )
        return dict(updated_trust)

    @abstractmethod
    def propose_deal(
        self,
        history: list[RoundRecord],
        round_idx: int,
        total_rounds: int,
        is_opening: bool,
        initial_deal: str | None,
        display_names: dict[str, str],
    ) -> tuple[Deal, str, str]:
        pass

    @abstractmethod
    def vote_on_deal(
        self,
        deal: Deal,
        proposer_id: str,
        history: list[RoundRecord],
        round_idx: int,
        total_rounds: int,
        display_names: dict[str, str],
        proposer_rationale: str,
    ) -> tuple[bool, str, str]:
        pass
