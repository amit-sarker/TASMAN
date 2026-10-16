from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from src.game.deal import format_deal
from src.agents.prompt_adapter import SUPPLEMENT_PROMPT_PROTOCOL
from src.utils.seeding import LONGITUDINAL_DESIGN_VERSION

class NegotiationLogger:
    def __init__(self, logging_config: Any, out_dir_override: str | None = None) -> None:
        if out_dir_override:
            self.run_dir = Path(out_dir_override)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            parent = Path(logging_config.out_dir)
            parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for collision_idx in range(1000):
                suffix = "" if collision_idx == 0 else f"_{collision_idx:03d}"
                candidate = parent / f"run_{timestamp}{suffix}"
                try:
                    candidate.mkdir(exist_ok=False)
                except FileExistsError:
                    continue
                self.run_dir = candidate
                break
            else:
                raise RuntimeError(
                    "Could not allocate a unique timestamped experiment directory"
                )
        self.log_full_prompts = logging_config.log_full_prompts
        self.log_full_completions = logging_config.log_full_completions
        self.console = getattr(logging_config, "console", True)
        self.console_private = getattr(logging_config, "console_private", True)
        self.rounds_path = self.run_dir / "rounds.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.incidents_path = self.run_dir / "incidents.jsonl"
        self.prompts_path = self.run_dir / "prompts.jsonl"
        self.trust_updates_path = self.run_dir / "trust_updates.jsonl"
        self.brs_feedback_path = self.run_dir / "brs_feedback.jsonl"
        self.brs_tools_path = self.run_dir / "brs_tools.jsonl"
        self.experiment_path = self.run_dir / "experiment.json"

        self._lock = threading.RLock()

        self._structured_output_pending: dict[
            str,
            dict[str, dict[str, int]],
        ] = {}

    @staticmethod
    def _structured_attempt_key(payload: dict[str, Any]) -> str:
        explicit = payload.get("attempt_id")
        if explicit and str(explicit) != "legacy":
            return str(explicit)
        trajectory = int(
            payload.get("trajectory_idx", payload.get("run_idx", 1)) or 1
        )
        episode = int(
            payload.get(
                "episode_position",
                payload.get("episode_idx", 0),
            )
            or 0
        )
        attempt = int(payload.get("attempt_idx", 1) or 1)
        return f"t{trajectory:03d}-e{episode:03d}-a{attempt:03d}"

    @staticmethod
    def _empty_structured_role_counts() -> dict[str, int]:
        return {
            "initial_valid": 0,
            "control_repaired": 0,
            "regenerated": 0,
            "reformatted": 0,
            "accepted_over_limit": 0,
            "unrecoverable": 0,
            "failed_reformat_attempts": 0,
            "failed_regeneration_attempts": 0,
            "regeneration_used_actions": 0,
        }

    @classmethod
    def _accumulate_structured_event(
        cls,
        by_role: dict[str, dict[str, int]],
        payload: dict[str, Any],
    ) -> None:
        role = str(payload.get("role", "") or "unknown")
        counts = by_role.setdefault(role, cls._empty_structured_role_counts())
        stage = str(payload.get("stage", ""))
        success = bool(payload.get("success"))
        terminal = bool(payload.get("terminal_outcome", True))
        terminal_key = {
            "initial": "initial_valid",
            "control_character_repair": "control_repaired",
            "regeneration": "regenerated",
            "reformat": "reformatted",
            "accepted_over_limit": "accepted_over_limit",
            "unrecoverable": "unrecoverable",
        }.get(stage)
        if terminal and terminal_key and (success or stage == "unrecoverable"):
            counts[terminal_key] += 1
        if stage == "reformat" and not success:
            counts["failed_reformat_attempts"] += 1
        if stage == "regeneration" and not success:
            counts["failed_regeneration_attempts"] += 1
        if stage == "regeneration" and success:
            counts["regeneration_used_actions"] += 1

    def _structured_output_summary_for_attempt(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        key = self._structured_attempt_key(payload)
        by_role = self._structured_output_pending.get(key)
        if not by_role:
            return None
        return {
            "schema_version": 1,
            "telemetry_protocol": "per_attempt_summary_v1",
            "by_role": {
                role: dict(values) for role, values in by_role.items()
            },
        }

    @staticmethod
    def _identity_payload(
        *,
        episode_idx: int,
        run_idx: int,
        identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        extra = {key: value for key, value in (identity or {}).items() if value is not None}
        trajectory_idx = int(extra.pop("trajectory_idx", run_idx) or run_idx)
        episode_position = int(extra.pop("episode_position", episode_idx) or episode_idx)
        if episode_position != int(episode_idx):
            raise ValueError("episode_idx must equal episode_position")
        payload: dict[str, Any] = {
            "design_version": extra.pop("design_version", LONGITUDINAL_DESIGN_VERSION),
            "trajectory_idx": trajectory_idx,
            "run_idx": trajectory_idx,
            "episode_position": episode_position,
            "episode_idx": episode_position,
        }
        for key in (
            "trajectory_seed", "episode_seed", "run_seed", "scenario_idx",
            "scenario_path", "scenario_hash", "negotiation_id", "attempt_idx",
            "attempt_id",
        ):
            if key in extra:
                payload[key] = extra.pop(key)
        payload.update(extra)
        return payload

    @staticmethod
    def _record_identity(record: Any, run_idx: int | None = None) -> dict[str, Any]:
        trajectory = int(
            getattr(record, "trajectory_idx", None)
            or getattr(record, "run_idx", None)
            or run_idx
            or 1
        )
        episode = int(getattr(record, "episode_idx", 0))
        episode_position = int(
            getattr(record, "episode_position", None) or episode
        )
        if episode_position != episode:
            raise ValueError("episode_idx must equal episode_position")
        fields = {
            "design_version": getattr(record, "design_version", LONGITUDINAL_DESIGN_VERSION),
            "trajectory_idx": trajectory,
            "run_idx": trajectory,
            "episode_position": episode_position,
            "episode_idx": episode_position,
        }
        for key in (
            "trajectory_seed", "episode_seed", "scenario_idx", "negotiation_id",
            "attempt_idx", "attempt_id",
        ):
            value = getattr(record, key, None)
            if value is not None and value != "":
                fields[key] = value
        return fields

    def _normalize_scoped_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "episode_idx" not in payload and "episode_position" not in payload:
            return payload
        episode = int(payload.get("episode_position", payload.get("episode_idx", 0)) or 0)
        trajectory = int(payload.get("trajectory_idx", payload.get("run_idx", 1)) or 1)
        extra = dict(payload)
        extra.pop("episode_idx", None)
        extra.pop("episode_position", None)
        extra.pop("run_idx", None)
        extra.pop("trajectory_idx", None)
        return self._identity_payload(
            episode_idx=episode,
            run_idx=trajectory,
            identity=extra,
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def log_round(
        self,
        *,
        episode_idx: int,
        run_idx: int = 1,
        record,
        total_accepts: int,
        min_total_accepts: int,
        display_names: dict[str, str] | None = None,
        deal_utilities_by_agent: dict[str, int] | None = None,
        meets_threshold_by_agent: dict[str, bool] | None = None,
        proposer_utility: int | None = None,
        proposer_vote_type: str = "",
        proposer_trust_in_voters: dict[str, float | None] | None = None,
        trust_visible: bool = True,
        trust_visibility_by_agent: dict[str, bool] | None = None,
        **identity: Any,
    ) -> None:
        issue_order = list(record.deal.choices)
        payload = {
            **self._identity_payload(
                episode_idx=episode_idx, run_idx=run_idx, identity=identity,
            ),
            "round_idx": record.round_idx,
            "proposer_id": record.proposer_id,
            "proposer_implicit_accept": True,
            "deal": record.deal.model_dump(),
            "deal_text": format_deal(record.deal, issue_order),
            "rationale": record.rationale,
            "votes": [vote.model_dump() for vote in record.votes],
            "total_accepts": total_accepts,
            "min_total_accepts": min_total_accepts,
            "passed": record.passed,
            "public_answers": record.public_answers,
            "scratchpads": record.scratchpads,
            "display_names": display_names or {},
            "trust_visible": trust_visible,
            "trust_visibility_by_agent": trust_visibility_by_agent or {},

            "_private_deal_utilities": deal_utilities_by_agent or {},
            "_private_meets_threshold": meets_threshold_by_agent or {},
            "_private_proposer_utility": proposer_utility,
            "_private_proposer_vote_type": proposer_vote_type,
            "_private_proposer_trust_in_voters": proposer_trust_in_voters or {},
        }
        with self._lock:
            self._append_jsonl(self.rounds_path, payload)
            if self.console:
                self._print_round(payload)

    def log_episode_summary(
        self,
        record,
        display_names: dict[str, str] | None = None,
        run_idx: int = 1,
        *,
        brs_state_before: dict[str, Any] | None = None,
        brs_state_after: dict[str, Any] | None = None,
        brs_feedback_key: str | None = None,
    ) -> None:
        display_names = display_names or {}
        deal_text = None
        if record.final_deal is not None:
            deal_text = format_deal(record.final_deal, list(record.final_deal.choices))
        payload = {
            **self._record_identity(record, run_idx),
            "scenario_path": getattr(record, "scenario_path", ""),
            "scenario_hash": getattr(record, "scenario_hash", ""),
            "difficulty": getattr(record, "difficulty", ""),
            "trust_mode": getattr(record, "trust_mode", ""),
            "trust_is_frozen": getattr(record, "trust_is_frozen", False),
            "prompt_version": getattr(record, "prompt_version", ""),
            "supplement_prompt_protocol": SUPPLEMENT_PROMPT_PROTOCOL,
            "run_seed": getattr(record, "run_seed", getattr(record, "episode_seed", 0)),
            "proposer_order": getattr(record, "proposer_order", []),
            "malicious_agent_id": record.malicious_agent_id,

            "agent_types": getattr(record, "agent_types", {}),
            "success": record.success,
            "final_deal": record.final_deal.model_dump() if record.final_deal else None,
            "final_deal_text": deal_text,
            "utilities": record.utilities,

            "_private_thresholds_by_agent": getattr(record, "thresholds_by_agent", {}),
            "trust_before": record.trust_before,
            "trust_after": record.trust_after,
            "num_rounds": len(record.rounds),
            "rounds_to_deal": getattr(record, "rounds_to_deal", None),
            "max_rounds": getattr(record, "max_rounds", 0),
            "n_agents": getattr(record, "n_agents", 0),
            "min_non_proposer_accepts": max(0, getattr(record, "n_agents", 0) - 2),
            "min_total_accepts": max(0, getattr(record, "n_agents", 0) - 1),
            "pass_rule": "n_minus_1_total",
            "trust_condition_enabled": getattr(
                record,
                "trust_condition_enabled",
                getattr(record, "trust_mode", "") != "none",
            ),
            "trust_visible": getattr(record, "trust_visible", False),
            "trust_visibility_by_agent": getattr(
                record,
                "trust_visibility_by_agent",
                {},
            ),
            "trust_visibility_policy": getattr(
                record,
                "trust_visibility_policy",
                "",
            ),
            "display_names": display_names,
        }
        for field in ("trust_replay", "trust_random"):
            value = getattr(record, field, None)
            if value:
                payload[field] = value
        for field, supplied in (
            ("brs_state_before", brs_state_before),
            ("brs_state_after", brs_state_after),
            ("brs_feedback_key", brs_feedback_key),
        ):
            value = supplied if supplied is not None else getattr(record, field, None)
            if supplied is not None or value not in (None, {}, ""):
                payload[field] = value
        with self._lock:
            self._append_jsonl(self.episodes_path, payload)
            if self.console:
                self._print_episode_summary(payload)

    def log_brs_feedback(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.brs_feedback_path, self._normalize_scoped_payload(dict(payload)),
        )

    def log_brs_tool(self, payload: dict[str, Any]) -> None:
        self._append_jsonl(
            self.brs_tools_path, self._normalize_scoped_payload(dict(payload)),
        )

    def log_experiment_summary(
        self,
        records,
        llm_usage: dict[str, Any],
        trust_mode: str = "",
        agent_models: dict[str, str] | None = None,
        scenario_metadata: dict[str, Any] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        records = list(records)
        settings = dict(run_config or {})
        settings["supplement_prompt_protocol"] = SUPPLEMENT_PROMPT_PROTOCOL
        identities = [self._record_identity(record) for record in records]
        trajectory_positions: dict[int, set[int]] = {}
        trajectory_seeds: dict[str, Any] = {}
        for identity in identities:
            trajectory = identity["trajectory_idx"]
            trajectory_positions.setdefault(trajectory, set()).add(identity["episode_position"])
            trajectory_seeds[str(trajectory)] = identity.get("trajectory_seed", 0)
        episode_positions = {identity["episode_position"] for identity in identities}
        configured_episodes = int(settings.get("num_episodes", len(episode_positions)))
        required_positions = set(range(1, configured_episodes + 1))
        completed_trajectories = sorted(
            trajectory for trajectory, positions in trajectory_positions.items()
            if required_positions and required_positions.issubset(positions)
        )
        payload = {
            "schema_version": 2,
            "raw_logs_only": True,
            "design_version": settings.get("design_version", LONGITUDINAL_DESIGN_VERSION),
            "supplement_prompt_protocol": SUPPLEMENT_PROMPT_PROTOCOL,
            "aggregation_unit": "trajectory",
            "num_episodes": len(episode_positions),
            "episodes_per_trajectory": configured_episodes,
            "num_trajectories": len(trajectory_positions),
            "num_runs_per_episode": len(trajectory_positions),
            "completed_trajectories": completed_trajectories,
            "trajectory_seeds": trajectory_seeds,
            "total_records": len(records),
            "trust_aggregation_used_for_execution": False,
            "trust_mode": trust_mode,
            "trust_condition_enabled": settings.get("trust_condition_enabled", trust_mode != "none"),
            "trust_is_frozen": settings.get("trust_is_frozen", False),
            "agent_models": agent_models or {},
            "scenario_metadata": scenario_metadata or {},
            "llm_usage": llm_usage,
            "llm_usage_cumulative": llm_usage,
            "llm_usage_current_segment": settings.get("llm_usage_current_segment", llm_usage),
            "llm_usage_prior_segments": settings.get("llm_usage_prior_segments", {}),
            "run_config": settings,
        }
        with self._lock:
            temporary: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", dir=self.run_dir,
                    prefix=".experiment-", suffix=".tmp", delete=False,
                ) as handle:
                    temporary = handle.name
                    json.dump(payload, handle, ensure_ascii=True, sort_keys=True,
                              indent=2, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.experiment_path)
                directory = os.open(self.run_dir, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if temporary is not None and os.path.exists(temporary):
                    os.unlink(temporary)

    def _print_round(self, payload: dict[str, Any]) -> None:
        print(
            f"  Episode {payload['episode_idx']} | "
            f"Trajectory {payload['trajectory_idx']} | Round {payload['round_idx']}"
        )

    def _print_episode_summary(self, payload: dict[str, Any]) -> None:
        outcome = "DEAL" if payload.get("success") else "NO DEAL"
        rounds = payload.get("rounds_to_deal") or payload.get("num_rounds", 0)
        print(
            f"  ✓ Trajectory {payload.get('trajectory_idx', payload.get('run_idx', 1))} "
            f"episode {payload.get('episode_idx', 0)} "
            f"complete: {outcome} ({rounds} round{'s' if rounds != 1 else ''})"
        )

    def log_trust_update(
        self,
        *,
        episode_idx: int,
        run_idx: int = 1,
        agent_id: str,
        scratchpad: str,
        scratchpad_by_agent: dict[str, str] | None = None,
        trust_scores_before: dict[str, float] | None = None,
        trust_scores_prompted: dict[str, float] | None = None,
        trust_scores_after: dict[str, float],
        display_names: dict[str, str] | None = None,
        trust_mode: str = "",
        model_name: str = "",
        is_malicious: bool = False,
        agent_type: str = "",
        episode_signals: dict[str, float] | None = None,
        metrics_weights: dict[str, float] | None = None,
        **identity: Any,
    ) -> None:
        payload: dict[str, Any] = {
            **self._identity_payload(
                episode_idx=episode_idx, run_idx=run_idx, identity=identity,
            ),
            "agent_id": agent_id,
            "trust_mode": trust_mode,
            "model_name": model_name,
            "is_malicious": is_malicious,
            "agent_type": agent_type,
            "scratchpad": scratchpad,
            "trust_scores_before": trust_scores_before or {},
            "trust_scores_after": trust_scores_after,
            "episode_signals": episode_signals or {},
            "metrics_weights": metrics_weights or {},
            "display_names": display_names or {},
        }
        if scratchpad_by_agent:
            payload["scratchpad_by_agent"] = scratchpad_by_agent
        if trust_scores_prompted is not None:
            payload["trust_scores_prompted"] = trust_scores_prompted
        self._append_jsonl(self.trust_updates_path, payload)

    def log_prompt_completion(
        self,
        *,
        episode_idx: int = 0,
        run_idx: int = 1,
        agent_id: str,
        round_idx: int,
        role: str,
        system_prompt: str,
        user_prompt: str,
        completion: str,
        raw_completion: str | None = None,
        retry: bool = False,
        recovery_stage: str | None = None,
        recovery_attempt: int = 0,
        provider: str | None = None,
        model: str | None = None,
        finish_reason: str | None = None,
        response_length: int | None = None,
        returned_length: int | None = None,
        scratchpad_length: int | None = None,
        scratchpad_max_target_length: int | None = None,
        scratchpad_recovery_target: int | None = None,
        scratchpad_recovery_total_target: int | None = None,
        candidate_tokens: int | None = None,
        thinking_tokens: int | None = None,
        reasoning_content_length: int | None = None,
        reasoning_content_present: bool | None = None,
        prompt_tokens: int | None = None,
        request_seed: int | None = None,
        structured_output: bool | None = None,
        mechanical_reformat: bool | None = None,
        thinking_mode: str | None = None,
        max_token_parameter: str | None = None,
        system_fingerprint: str | None = None,
        **identity: Any,
    ) -> None:
        if not self.log_full_prompts and not self.log_full_completions:
            return
        payload: dict[str, Any] = {
            **self._identity_payload(
                episode_idx=episode_idx, run_idx=run_idx, identity=identity,
            ),
            "agent_id": agent_id,
            "round_idx": round_idx,
            "role": role,
            "retry": retry,
        }
        if self.log_full_prompts:
            payload["system_prompt"] = system_prompt
            payload["user_prompt"] = user_prompt
        if self.log_full_completions:
            payload["completion"] = completion
            if raw_completion is not None:
                payload["raw_completion"] = raw_completion
        optional_fields = {
            "recovery_stage": recovery_stage,
            "recovery_attempt": recovery_attempt if recovery_stage else None,
            "provider": provider,
            "model": model,
            "finish_reason": finish_reason,
            "response_length": response_length,
            "returned_length": returned_length,
            "scratchpad_length": scratchpad_length,
            "scratchpad_max_target_length": scratchpad_max_target_length,
            "scratchpad_recovery_target": scratchpad_recovery_target,
            "scratchpad_recovery_total_target": (
                scratchpad_recovery_total_target
            ),
            "candidate_tokens": candidate_tokens,
            "thinking_tokens": thinking_tokens,
            "reasoning_content_length": reasoning_content_length,
            "reasoning_content_present": reasoning_content_present,
            "prompt_tokens": prompt_tokens,
            "request_seed": request_seed,
            "structured_output": structured_output,
            "mechanical_reformat": mechanical_reformat,
            "thinking_mode": thinking_mode,
            "max_token_parameter": max_token_parameter,
            "system_fingerprint": system_fingerprint,
        }
        payload.update(
            {key: value for key, value in optional_fields.items() if value is not None}
        )
        self._append_jsonl(self.prompts_path, payload)

    def log_privacy_violation(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "privacy_violation", **payload}),
        )

    def log_parse_failure(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "parse_failure", **payload}),
        )

    def log_structured_output_recovery(self, **payload: Any) -> None:
        normalized = self._normalize_scoped_payload(
            {"type": "structured_output_recovery", **payload}
        )
        normalized["telemetry_protocol"] = "per_attempt_summary_v1"
        routine_initial_success = (
            normalized.get("stage") == "initial"
            and bool(normalized.get("success"))
            and bool(normalized.get("terminal_outcome", True))
            and not normalized.get("error")
        )
        with self._lock:
            key = self._structured_attempt_key(normalized)
            by_role = self._structured_output_pending.setdefault(key, {})
            self._accumulate_structured_event(by_role, normalized)
            if not routine_initial_success:
                self._append_jsonl(self.incidents_path, normalized)

    def log_vote_inconsistency(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "vote_inconsistency", **payload}),
        )

    def log_content_filter(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "content_filter", **payload}),
        )

    def log_malicious_strategic_accept(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "malicious_strategic_accept", **payload}),
        )

    def log_trust_update_frozen(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "trust_update_frozen", **payload}),
        )

    def log_run_started(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "run_started", **payload}),
        )

    def log_run_completed(self, **payload: Any) -> None:
        normalized = self._normalize_scoped_payload(
            {"type": "run_completed", **payload}
        )
        with self._lock:
            summary = self._structured_output_summary_for_attempt(normalized)
            if summary is not None:
                normalized["structured_output_summary"] = summary
            self._append_jsonl(self.incidents_path, normalized)
            if summary is not None:
                self._structured_output_pending.pop(
                    self._structured_attempt_key(normalized),
                    None,
                )

    def log_run_failure(self, **payload: Any) -> None:
        normalized = self._normalize_scoped_payload(
            {"type": "run_failure", **payload}
        )
        with self._lock:
            summary = self._structured_output_summary_for_attempt(normalized)
            if summary is not None:
                normalized["structured_output_summary"] = summary
            self._append_jsonl(self.incidents_path, normalized)
            if summary is not None:
                self._structured_output_pending.pop(
                    self._structured_attempt_key(normalized),
                    None,
                )

    def log_trajectory_cell_retry(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload(
                {"type": "trajectory_cell_retry", **payload}
            ),
        )

    def log_trajectory_cell_recovered(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload(
                {"type": "trajectory_cell_recovered", **payload}
            ),
        )

    def log_run_incomplete(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            {"type": "run_incomplete", **payload},
        )

    def log_above_threshold_reject(self, **payload: Any) -> None:
        self._append_jsonl(
            self.incidents_path,
            self._normalize_scoped_payload({"type": "above_threshold_reject", **payload}),
        )
