from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.agents.honest_agent import HonestAgent, THRESHOLD_ACTION_POLICY
from src.agents.malicious_agent import MaliciousAgent
from src.agents.prompt_adapter import SUPPLEMENT_PROMPT_PROTOCOL
from src.agents.structured_output import StructuredOutputError
from src.agents.stub_client import StubLLMClient
from src.BRS.brs import (
    initialize_observer_state, serialize_observer_state, deserialize_observer_state,
    normalize_counts, update_brs, scores_from_observer_state,
)
from src.BRS.logging import log_saved_call
from src.BRS.feedback import FeedbackError, parse_agent_feedback_output, parse_feedback_output
from src.BRS.state_store import BRSStateStore, StateStoreError
from src.BRS.recovery import (output_correction_error, saved_feedback_is_blocked,
                              transport_retry_error)
from src.BRS.tool import counterpart_scores
from src.game.episode import Episode
from src.game.scenario import Scenario
from src.game.trust_signals_v4 import V4_TRUST_UPDATE_PROTOCOL
from src.prompts.prompt_builder import format_history
from src.trust.trust_calculator import TrustCalculator
from src.trust.replay import TrustReplaySchedule
from src.trust.random_schedule import RANDOM_TRUST_PROTOCOL, RandomTrustSchedule
from src.utils.seeding import (
    LONGITUDINAL_DESIGN_VERSION,
    SEED_SCHEME_VERSION,
    derive_episode_seed,
    derive_trajectory_seed,
)
from src.utils.types import EpisodeRecord, TrustScores

class _EpisodeSummaryProxy:
    __slots__ = (
        "episode_idx", "episode_position", "run_idx", "trajectory_idx",
        "trajectory_seed", "episode_seed", "scenario_idx", "scenario_path",
        "scenario_hash", "difficulty", "trust_mode", "run_seed",
        "design_version", "negotiation_id", "attempt_idx", "attempt_id",
        "malicious_agent_id",
        "success", "rounds", "rounds_to_deal", "utilities",
    )

    def __init__(self, row: dict[str, Any]) -> None:
        self.episode_position: int = int(
            row.get("episode_position", row.get("episode_idx", 0))
        )
        self.episode_idx: int = int(row.get("episode_idx", self.episode_position))
        if self.episode_idx != self.episode_position:
            raise ValueError("episode_idx must equal episode_position")
        self.run_idx: int = row.get("run_idx", row.get("trajectory_idx", 1))
        self.trajectory_idx: int = row.get("trajectory_idx", self.run_idx)
        self.trajectory_seed: int = row.get("trajectory_seed", 0)
        self.episode_seed: int = row.get("episode_seed", row.get("run_seed", 0))
        self.scenario_idx: int = row.get("scenario_idx", 0)
        self.scenario_path: str = row.get("scenario_path", "")
        self.scenario_hash: str = row.get("scenario_hash", "")
        self.difficulty: str = row.get("difficulty", "")
        self.trust_mode: str = row.get("trust_mode", "")
        self.run_seed: int = row.get("run_seed", 0)
        self.design_version: str = row.get("design_version", "")
        self.negotiation_id: str = row.get(
            "negotiation_id",
            f"t{self.trajectory_idx:03d}-e{self.episode_position:03d}",
        )
        self.attempt_idx: int = row.get("attempt_idx", 1)
        self.attempt_id: str = row.get(
            "attempt_id", f"{self.negotiation_id}-a{self.attempt_idx:03d}"
        )
        self.malicious_agent_id: str = row.get("malicious_agent_id", "")
        self.success: bool = bool(row.get("success", False))

        self.rounds: list[None] = [None] * int(row.get("num_rounds", 0))
        self.rounds_to_deal: int | None = row.get("rounds_to_deal")
        self.utilities: dict[str, int] = row.get("utilities", {})

class GameEngine:
    def __init__(
        self,
        game_config: Any,
        scenarios: list[Scenario],
        llm_client_factory: Callable[[str | None, Any], Any],
        logger: Any,
        resume: bool = False,
        resume_force: bool = False,
        retry_brs_feedback: list[str] | None = None,
    ) -> None:
        self._resume = resume
        self._resume_force = resume_force
        self._prior_llm_usage: dict[str, int] = self._empty_usage()

        self._resume_start_episode: int = 0
        if not scenarios:
            raise ValueError("GameEngine requires at least one scenario.")
        self.config = game_config
        self._trust_is_frozen = self.config.trust.is_frozen
        self._replay_config = getattr(self.config.trust, "replay", None)
        self._trust_replay: TrustReplaySchedule | None = None
        self._random_each_episode = getattr(self.config.trust, "random_each_episode", False)
        self._trust_random: RandomTrustSchedule | None = None
        self._trust_schedule: TrustReplaySchedule | RandomTrustSchedule | None = None
        if self._replay_config and (self._trust_is_frozen or self.config.trust.mode == "none"):
            raise ValueError("Trust replay cannot be combined with frozen or disabled trust")
        if self._random_each_episode and (
            self._trust_is_frozen or self._replay_config is not None or self.config.trust.mode == "none"
        ):
            raise ValueError("Random trust cannot be combined with frozen, replayed, or disabled trust")

        self._is_brs = (self.config.trust.mode == "brs"
                        and not self._trust_is_frozen and self._replay_config is None
                        and not self._random_each_episode)
        self._scenarios = scenarios
        self._retry_brs_feedback: set[tuple[str, str]] = set()
        self._consumed_brs_retries: set[tuple[str, str]] = set()
        if retry_brs_feedback and (not resume or not self._is_brs):
            raise ValueError("--retry-brs-feedback requires --resume and unfrozen trust.mode=brs")
        for selector in retry_brs_feedback or []:
            match = re.fullmatch(r"t([0-9]+)-e([0-9]+):(.+)", selector)
            if (not match or int(match[1]) < 1 or int(match[2]) < 1
                    or match[3] not in scenarios[0].all_agent_ids()):
                raise ValueError(f"Invalid --retry-brs-feedback selector: {selector!r}; use t002-e001:agent_id")
            key = BRSStateStore.key(int(match[1]), int(match[2]))
            self._retry_brs_feedback.add((key, match[3]))

        self._scenario_paths = [
            getattr(scenario, "_source_path", "") for scenario in scenarios
        ]
        self._scenario_hashes = [
            self._scenario_hash(path) for path in self._scenario_paths
        ]
        self.logger = logger
        self._guard_fresh_run_directory()
        self._base_seed = self.config.game.base_seed

        self._llm_client_factory = llm_client_factory

        self._is_v4_protocol = (
            str(getattr(self.config.prompts, "prompt_version", "") or "") == "v4"
        )

        self._oracle_unavailable_metric_default = 0.0
        self._oracle = TrustCalculator(
            self.config.trust,
            unavailable_metric_default=self._oracle_unavailable_metric_default,
        )

        first_ids = set(scenarios[0].all_agent_ids())
        for i, sc in enumerate(scenarios[1:], start=2):
            if set(sc.all_agent_ids()) != first_ids:
                raise ValueError(
                    f"Scenario {i} has different agent IDs than scenario 1. "
                    "All scenarios must define the same agents."
                )

        expected_malicious = (
            self.config.malicious.agent_id
            if self.config.malicious.agent_id is not None
            else scenarios[0].all_agent_ids()[-1]
        )
        if not self.config.malicious.enabled:
            raise ValueError("The supplement experiment requires a strategic malicious agent")
        if expected_malicious not in first_ids:
            raise ValueError("malicious.agent_id must identify a participant in every scenario")
        for i, sc in enumerate(scenarios, start=1):
            for aid, cfg in sc.agent_configs.items():
                atype = getattr(cfg, "agent_type", "") or ""
                if atype.startswith("malicious") and aid != expected_malicious:
                    raise ValueError(
                        f"Scenario {i} assigns agent_type '{atype}' to '{aid}', but the "
                        f"engine runs '{expected_malicious}' as the malicious agent "
                        "(malicious.agent_id). Fix the scenario or the config."
                    )
                if aid == expected_malicious and atype and not atype.startswith("malicious"):
                    raise ValueError(
                        f"Scenario {i} assigns honest agent_type '{atype}' to "
                        f"'{aid}', which the engine runs as the malicious agent."
                    )

        if self._replay_config is not None:
            run_dir = getattr(self.logger, "run_dir", None)
            if run_dir is None:
                raise ValueError("Trust replay requires a durable experiment output directory")

            self._trust_replay = TrustReplaySchedule.load(
                self._replay_config.source_run,
                config=self.config,
                expected_cells=self._replay_expected_cells(),
                output_dir=run_dir,
                resume=self._resume,
            )
        if self._random_each_episode:
            self._trust_random = RandomTrustSchedule(
                scenarios[0].all_agent_ids(), self._base_seed,
            )
        self._trust_schedule = self._trust_replay or self._trust_random

        agent_models: dict[str, str] = getattr(game_config, "agent_models", {}) or {}
        agent_endpoints: dict = getattr(game_config, "agent_endpoints", {}) or {}
        self._agent_clients: dict[str, Any] = {}
        self._agent_client_provenance: dict[str, dict[str, Any]] = {}

        self._agent_model_names: dict[str, str] = {}
        for agent_id, agent_config in scenarios[0].agent_configs.items():
            model_override: str | None = (
                agent_models.get(agent_id)
                or getattr(agent_config, "model", None)
            )
            self._agent_model_names[agent_id] = model_override or ""
            endpoint_cfg = agent_endpoints.get(agent_id)
            client = llm_client_factory(model_override, endpoint_cfg)
            self._agent_clients[agent_id] = client
            self._agent_client_provenance[agent_id] = self._describe_client(
                client,
                configured_model=model_override or "",
            )

        if self._trust_replay is not None:
            self._trust_replay.validate_client_provenance(
                self._agent_client_provenance,
                allow_stub=self.config.llm.dry_run,
            )

        if self._is_brs:
            if self.config.trust.initial_score != 0.5 or self.config.trust.brs is None:
                raise ValueError("BRS requires its configuration and a fixed 0.5 initial score")
            if getattr(self.logger, "run_dir", None) is None:
                raise ValueError("BRS requires a durable experiment output directory")
            self._brs_state = {
                idx: initialize_observer_state(scenarios[0].all_agent_ids())
                for idx in range(1, self._trajectory_count() + 1)
            }
            self._brs_store = BRSStateStore(self.logger.run_dir)
            self._usage_segment = uuid.uuid4().hex

    def _replay_expected_cells(self) -> dict[tuple[int, int], dict[str, Any]]:
        cells = {}
        for trajectory_idx in range(1, self._trajectory_count() + 1):
            trajectory_seed = derive_trajectory_seed(self._base_seed, trajectory_idx)
            for episode_idx in range(1, self.config.game.num_episodes + 1):
                scenario_idx = self._scenario_index(trajectory_idx, episode_idx)
                scenario = self._scenarios[scenario_idx - 1]
                scenario_hash = self._scenario_hashes[scenario_idx - 1]
                episode_seed = derive_episode_seed(
                    trajectory_seed, episode_idx,
                    self._scenario_identity(scenario_idx,
                        self._scenario_paths[scenario_idx - 1], scenario_hash),
                )
                malicious_id = self._select_malicious_agent(episode_idx - 1)
                proposer_order = list(scenario.all_agent_ids())
                random.Random(episode_seed).shuffle(proposer_order)
                cells[trajectory_idx, episode_idx] = {
                    "trajectory_idx": trajectory_idx,
                    "episode_position": episode_idx,
                    "trajectory_seed": trajectory_seed,
                    "episode_seed": episode_seed,
                    "scenario_idx": scenario_idx,
                    "scenario_hash": scenario_hash,
                    "malicious_agent_id": malicious_id,
                    "agent_types": {
                        aid: "malicious_strategic" if aid == malicious_id else cfg.agent_type or ""
                        for aid, cfg in scenario.agent_configs.items()
                    },
                    "proposer_order": proposer_order,
                    "max_rounds": len(proposer_order) * self.config.game.rounds_per_episode_multiplier,
                    "_private_thresholds_by_agent": {
                        aid: (self.config.malicious.threshold if aid == malicious_id else cfg.min_threshold)
                        for aid, cfg in scenario.agent_configs.items()
                    },
                }
        return cells

    def _validate_scheduled_trust_record(self, row: dict[str, Any]) -> None:
        if self._trust_schedule is None:
            return
        trajectory_idx = row["trajectory_idx"]
        episode_idx = row["episode_position"]
        expected = self._trust_schedule.snapshot(trajectory_idx, episode_idx)
        label = "Trust replay" if self._trust_replay is not None else "Random trust"
        field = "trust_replay" if self._trust_replay is not None else "trust_random"
        if row.get("trust_before") != expected or row.get("trust_after") != expected:
            raise RuntimeError(
                f"{label} mismatch for trajectory {trajectory_idx}, episode {episode_idx}"
            )
        if row.get(field) != self._trust_schedule.provenance(trajectory_idx, episode_idx):
            raise RuntimeError(f"{label} episode provenance differs from the expected schedule")

    def _guard_fresh_run_directory(self) -> None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return
        root = Path(run_dir)
        if self._resume:
            checkpoint_path = root / "checkpoint.json"
            if not checkpoint_path.is_file():
                raise RuntimeError(
                    f"Resume refused: output directory '{root}' has no "
                    "checkpoint.json. Choose the correct prior run directory, "
                    "or start fresh in a new --out-dir."
                )
            return
        artifact_names = {
            "rounds.jsonl",
            "episodes.jsonl",
            "incidents.jsonl",
            "prompts.jsonl",
            "trust_updates.jsonl",
            "averaged_trust.jsonl",
            "oracle_trust.jsonl",
            "trust_summary.jsonl",
            "summary.csv",
            "trajectory_summary.csv",
            "scenario_summary.csv",
            "experiment.json",
            "checkpoint.json",
            "brs_feedback.jsonl",
            "brs_tools.jsonl",
            "feedback_report.html",
            "run_status.json",
            "trust_replay.json",
        }
        existing = sorted(
            path.name
            for path in root.iterdir()
            if (path.name in artifact_names or path.name == "Replay"
                or (path.name == "brs_state" and path.is_dir()
                    and any(item.name != ".run.lock" for item in path.iterdir())))
        ) if root.exists() else []
        if existing:
            preview = ", ".join(existing[:5])
            if len(existing) > 5:
                preview += f", … (+{len(existing) - 5} more)"
            raise RuntimeError(
                f"Fresh run refused: output directory '{root}' already contains "
                f"experiment artifacts ({preview}). Use --resume to continue that "
                "experiment, or choose a new --out-dir."
            )

    def run(self) -> list[EpisodeRecord]:
        if self._is_brs:
            with self._brs_store.run_lock():
                if not self._resume and (Path(self.logger.run_dir) / "checkpoint.json").exists():
                    raise StateStoreError("BRS output has already been used; resume or choose a new directory")
                self._brs_report_ready = False
                try:
                    return self._run_experiment()
                finally:
                    if self._brs_report_ready:
                        self._write_brs_feedback_report()
        return self._run_experiment()

    def _write_brs_feedback_report(self) -> None:
        try:
            from src.BRS.feedback_report import write_feedback_report
            report_path = write_feedback_report(self.logger.run_dir)
            if getattr(self.logger, "console", True):
                print(f"  → Feedback review written: {report_path}")
        except Exception as exc:
            print(f"WARNING: Could not write BRS feedback report: {exc}", file=sys.stderr)

    def _run_experiment(self) -> list[EpisodeRecord]:
        num_trajectories = self._trajectory_count()
        self._validate_seed_plan()
        num_workers: int = max(1, getattr(self.config.game, "parallel_workers", 1) or 1)
        num_workers = min(num_workers, num_trajectories)

        initial_trust = TrustScores.initialize(
            self._scenarios[0].all_agent_ids(),
            self.config.trust.initial_score,
        )
        trust_by_trajectory = {
            trajectory_idx: TrustScores.from_snapshot(
                self._trust_schedule.snapshot(trajectory_idx, 1)
                if self._trust_schedule is not None else initial_trust.snapshot()
            )
            for trajectory_idx in range(1, num_trajectories + 1)
        }
        progress_by_trajectory = {
            trajectory_idx: 0 for trajectory_idx in range(1, num_trajectories + 1)
        }
        attempt_counts: dict[int, dict[int, int]] = {
            trajectory_idx: {} for trajectory_idx in range(1, num_trajectories + 1)
        }
        all_records: list[EpisodeRecord] = []
        exhausted_trajectories: set[int] = set()
        cell_retry_attempts = self._gemini_cell_retry_attempts()

        if self._resume:
            checkpoint = self._load_checkpoint()
            if checkpoint is not None:
                (
                    progress_by_trajectory,
                    trust_by_trajectory,
                    attempt_counts,
                    ckpt_data,
                ) = checkpoint
                if "run_config_fingerprint" in ckpt_data:
                    self._validate_checkpoint_fingerprint(
                        ckpt_data["run_config_fingerprint"],
                    )
                else:
                    raise RuntimeError("Supplement resume requires a checkpoint with configuration provenance")
                self._prior_llm_usage = self._normalise_usage(
                    ckpt_data.get("llm_usage_cumulative", {})
                )

                if self._is_brs:
                    self._brs_store.recover_usage()
                    self._prior_llm_usage = self._brs_store.prior_usage(
                        self._usage_segment
                    )
                self._validate_checkpoint_scenario_plan(ckpt_data)
                self._preflight_brs_feedback_retries(progress_by_trajectory)
                self._reconcile_checkpoint_with_episode_log(
                    progress_by_trajectory,
                    trust_by_trajectory,
                    attempt_counts,
                )
                for scores in trust_by_trajectory.values():
                    self._validate_frozen_trust(scores.snapshot())
                committed_episode = min(progress_by_trajectory.values())
                self._resume_start_episode = committed_episode
                all_records = self._load_prior_records(progress_by_trajectory)
                if getattr(self.logger, "console", True):
                    print(
                        f"\nResuming from checkpoint: "
                        f"all trajectories complete through episode "
                        f"{committed_episode}; retrying from episode "
                        f"{committed_episode + 1}."
                    )
            else:
                raise RuntimeError(
                    "Resume requested but checkpoint.json could not be loaded; "
                    "refusing to append to the existing run directory."
                )

        if self._is_brs:
            self._brs_report_ready = True
        start_episode = min(progress_by_trajectory.values()) + 1
        for episode_idx in range(start_episode, self.config.game.num_episodes + 1):
            pending = [
                trajectory_idx
                for trajectory_idx in range(1, num_trajectories + 1)
                if (
                    trajectory_idx not in exhausted_trajectories
                    and progress_by_trajectory[trajectory_idx] < episode_idx
                )
            ]
            if not pending:
                continue
            lagging = [
                trajectory_idx
                for trajectory_idx in pending
                if progress_by_trajectory[trajectory_idx] != episode_idx - 1
            ]
            if lagging:
                raise RuntimeError(
                    "Checkpoint contains non-contiguous trajectory progress at "
                    f"episode {episode_idx}: {lagging}"
                )

            if getattr(self.logger, "console", True):
                workers_note = (
                    f"  ({len(pending)} trajectories · {num_workers} workers)"
                    if num_workers > 1
                    else f"  ({len(pending)} trajectories)"
                )
                policy = getattr(self.config.game, "scenario_order_policy", "fixed")
                print(f"\nEpisode {episode_idx}{workers_note}  —  order={policy}")

            work_items: list[dict[str, Any]] = []
            for trajectory_idx in pending:
                scenario_idx = self._scenario_index(trajectory_idx, episode_idx)
                scenario = self._scenarios[scenario_idx - 1]
                scenario_label = self._scenario_paths[scenario_idx - 1]
                scenario_hash = self._scenario_hashes[scenario_idx - 1]
                trajectory_seed = derive_trajectory_seed(
                    self._base_seed, trajectory_idx
                )
                episode_seed = derive_episode_seed(
                    trajectory_seed,
                    episode_idx,
                    self._scenario_identity(scenario_idx, scenario_label, scenario_hash),
                )
                attempt_idx = attempt_counts[trajectory_idx].get(episode_idx, 0) + 1
                if self._is_brs:
                    saved = self._brs_store.load(self._brs_store.key(trajectory_idx, episode_idx))
                    if saved is not None:
                        attempt_idx = int(saved["record"]["attempt_idx"])
                attempt_counts[trajectory_idx][episode_idx] = attempt_idx
                negotiation_id = f"t{trajectory_idx:03d}-e{episode_idx:03d}"
                attempt_id = f"{negotiation_id}-a{attempt_idx:03d}"
                total_rounds = (
                    len(scenario.all_agent_ids())
                    * self.config.game.rounds_per_episode_multiplier
                )
                work_items.append({
                    "ep_num": episode_idx,
                    "episode_idx": episode_idx,
                    "run_num": trajectory_idx,
                    "run_seed": episode_seed,
                    "trajectory_idx": trajectory_idx,
                    "trajectory_seed": trajectory_seed,
                    "episode_seed": episode_seed,
                    "episode_position": episode_idx,
                    "scenario_idx": scenario_idx,
                    "attempt_idx": attempt_idx,
                    "attempt_id": attempt_id,
                    "trust_snapshot": (
                        self._trust_schedule.snapshot(trajectory_idx, episode_idx)
                        if self._trust_schedule is not None
                        else trust_by_trajectory[trajectory_idx].snapshot()
                    ),
                    "scenario": scenario,
                    "scenario_label": scenario_label,
                    "scenario_hash": scenario_hash,
                    "malicious_id": self._select_malicious_agent(episode_idx - 1),
                    "total_rounds": total_rounds,
                    "display_names": {
                        party.id: party.display_name for party in scenario.parties
                    },
                })

            self._write_checkpoint(
                progress_by_trajectory,
                trust_by_trajectory,
                attempt_counts,
                exhausted_trajectories=exhausted_trajectories,
            )

            results: dict[
                int,
                tuple[EpisodeRecord, dict[str, dict[str, float]]],
            ] = {}
            terminal_failures: dict[
                int,
                tuple[dict[str, Any], Exception, str],
            ] = {}
            failed_attempts_this_cell: dict[int, int] = {
                int(item["trajectory_idx"]): 0 for item in work_items
            }
            retry_batch = work_items
            while retry_batch:
                batch_results, batch_failures = self._execute_trajectory_batch(
                    retry_batch,
                    num_workers,
                )
                results.update(batch_results)
                next_batch: list[dict[str, Any]] = []
                retrying: set[int] = set()
                for trajectory_idx in sorted(batch_failures):
                    item, exc, traceback_text = batch_failures[trajectory_idx]
                    retryable = isinstance(exc, StructuredOutputError)
                    failures_so_far = failed_attempts_this_cell[trajectory_idx]
                    can_retry = (
                        retryable and failures_so_far < cell_retry_attempts
                    )
                    disposition = (
                        "scheduled"
                        if can_retry
                        else "exhausted"
                        if retryable and cell_retry_attempts > 0
                        else "fatal"
                    )
                    self._log_run_failure(
                        item,
                        exc,
                        traceback_text=traceback_text,
                        retry_disposition=disposition,
                        cell_retry_attempts=cell_retry_attempts,
                    )
                    if not can_retry:
                        terminal_failures[trajectory_idx] = (
                            item,
                            exc,
                            traceback_text,
                        )
                        continue

                    failed_attempts_this_cell[trajectory_idx] += 1
                    retry_item = self._next_cell_attempt(
                        item,
                        attempt_counts,
                    )
                    next_batch.append(retry_item)
                    retrying.add(trajectory_idx)
                    if hasattr(self.logger, "log_trajectory_cell_retry"):
                        self.logger.log_trajectory_cell_retry(
                            episode_idx=episode_idx,
                            run_idx=trajectory_idx,
                            failed_attempt_idx=int(item["attempt_idx"]),
                            retry_attempt_idx=int(retry_item["attempt_idx"]),
                            retry_number=failed_attempts_this_cell[trajectory_idx],
                            retry_limit=cell_retry_attempts,
                            error=str(exc),
                            **self._identity_fields(
                                trajectory_idx=trajectory_idx,
                                trajectory_seed=int(item["trajectory_seed"]),
                                episode_idx=episode_idx,
                                scenario_idx=int(item["scenario_idx"]),
                                scenario_path=str(item["scenario_label"]),
                                scenario_hash=str(item["scenario_hash"]),
                                episode_seed=int(item["episode_seed"]),
                                attempt_idx=int(retry_item["attempt_idx"]),
                                attempt_id=str(retry_item["attempt_id"]),
                            ),
                        )
                if next_batch:
                    self._write_checkpoint(
                        progress_by_trajectory,
                        trust_by_trajectory,
                        attempt_counts,
                        retrying_trajectories=retrying,
                        exhausted_trajectories=exhausted_trajectories,
                    )
                retry_batch = next_batch

            episode_records: list[EpisodeRecord] = []
            for trajectory_idx in sorted(results):
                record, trust_snapshot = results[trajectory_idx]
                trust_by_trajectory[trajectory_idx] = TrustScores.from_snapshot(
                    trust_snapshot
                )
                progress_by_trajectory[trajectory_idx] = episode_idx
                if self._is_brs:
                    self._restore_brs_state(trajectory_idx, record.brs_state_after, trust_snapshot)
                episode_records.append(record)
                if (
                    failed_attempts_this_cell.get(trajectory_idx, 0) > 0
                    and hasattr(self.logger, "log_trajectory_cell_recovered")
                ):
                    self.logger.log_trajectory_cell_recovered(
                        episode_idx=episode_idx,
                        run_idx=trajectory_idx,
                        retry_count=failed_attempts_this_cell[trajectory_idx],
                        attempt_idx=int(record.attempt_idx),
                        attempt_id=str(record.attempt_id),
                        trajectory_idx=trajectory_idx,
                        trajectory_seed=int(record.trajectory_seed),
                        episode_position=episode_idx,
                        episode_seed=int(record.episode_seed),
                        negotiation_id=str(record.negotiation_id),
                        scenario_idx=int(record.scenario_idx),
                        scenario_path=str(record.scenario_path),
                        scenario_hash=str(record.scenario_hash),
                        design_version=LONGITUDINAL_DESIGN_VERSION,
                    )

            self._write_checkpoint(
                progress_by_trajectory,
                trust_by_trajectory,
                attempt_counts,
                failed_trajectories={
                    trajectory_idx
                    for trajectory_idx, (_item, exc, _tb) in terminal_failures.items()
                    if not (
                        isinstance(exc, StructuredOutputError)
                        and cell_retry_attempts > 0
                    )
                },
                exhausted_trajectories=(
                    exhausted_trajectories
                    | {
                        trajectory_idx
                        for trajectory_idx, (
                            _item,
                            exc,
                            _tb,
                        ) in terminal_failures.items()
                        if (
                            isinstance(exc, StructuredOutputError)
                            and cell_retry_attempts > 0
                        )
                    }
                ),
            )

            all_records.extend(episode_records)

            fatal_failures = {
                trajectory_idx: failure
                for trajectory_idx, failure in terminal_failures.items()
                if not (
                    isinstance(failure[1], StructuredOutputError)
                    and cell_retry_attempts > 0
                )
            }
            exhausted_now = set(terminal_failures) - set(fatal_failures)
            exhausted_trajectories.update(exhausted_now)
            if fatal_failures:
                raise RuntimeError(
                    "Episode "
                    f"{episode_idx}: trajectory failures {sorted(fatal_failures)}. "
                    "Successful trajectory states were checkpointed; resume will "
                    "retry only the failed cells. Check incidents.jsonl for details."
                )

        incomplete_trajectories = {
            trajectory_idx
            for trajectory_idx, progress in progress_by_trajectory.items()
            if progress < self.config.game.num_episodes
        }
        run_status = "incomplete" if incomplete_trajectories else "complete"
        self._write_run_status(
            status=run_status,
            progress_by_trajectory=progress_by_trajectory,
            attempt_counts=attempt_counts,
            incomplete_trajectories=incomplete_trajectories,
        )
        if incomplete_trajectories:
            if hasattr(self.logger, "log_run_incomplete"):
                self.logger.log_run_incomplete(
                    incomplete_trajectories=sorted(incomplete_trajectories),
                    progress_by_trajectory={
                        str(key): value
                        for key, value in sorted(progress_by_trajectory.items())
                    },
                    num_episodes=self.config.game.num_episodes,
                    retry_attempts_per_cell=cell_retry_attempts,
                )
            self._invalidate_incomplete_aggregates(incomplete_trajectories)
            print(
                "WARNING: run finished with incomplete trajectories "
                f"{sorted(incomplete_trajectories)}. Their trust state was not "
                "advanced, and final analysis was not generated. Resume this "
                "output directory to retry only the missing cells.",
                file=sys.stderr,
            )
            raise RuntimeError(
                "Run incomplete after exhausting automatic structured-output "
                "recovery for "
                f"trajectories {sorted(incomplete_trajectories)}. Healthy "
                "trajectories finished and were checkpointed; resume retries "
                "only the missing cells."
            )

        if hasattr(self.logger, "log_experiment_summary"):
            self.logger.log_experiment_summary(
                all_records,
                llm_usage=self._cumulative_usage(),
                trust_mode=self.config.trust.mode,
                agent_models=dict(self._agent_model_names),
                scenario_metadata=self._build_scenario_metadata(),
                run_config=self._build_run_config(),
            )

        self._write_trajectory_replays()

        self._write_analysis_html()

        return all_records

    def _run_single(
        self,
        *,
        episode_idx: int | None = None,
        run_num: int,
        run_seed: int,
        trajectory_idx: int,
        trajectory_seed: int,
        episode_seed: int,
        ep_num: int | None = None,
        episode_position: int | None = None,
        scenario_idx: int,
        attempt_idx: int,
        attempt_id: str,
        trust_snapshot: dict[str, dict[str, float]],
        scenario: Scenario,
        scenario_label: str,
        scenario_hash: str,
        malicious_id: str,
        total_rounds: int,
        display_names: dict[str, str],
    ) -> tuple[EpisodeRecord, dict[str, dict[str, float]]]:
        supplied_episode_ids = [
            int(value)
            for value in (episode_idx, ep_num, episode_position)
            if value is not None
        ]
        if not supplied_episode_ids:
            raise ValueError("An episode identifier is required")
        if len(set(supplied_episode_ids)) != 1:
            raise ValueError("episode_idx must equal episode_position and ep_num")
        episode_idx = supplied_episode_ids[0]
        episode_position = episode_idx

        if hasattr(self.logger, "log_run_started"):
            self.logger.log_run_started(
                episode_idx=episode_idx,
                run_idx=run_num,
                run_seed=run_seed,
                **self._identity_fields(
                    trajectory_idx=trajectory_idx,
                    trajectory_seed=trajectory_seed,
                    episode_idx=episode_idx,
                    scenario_idx=scenario_idx,
                    scenario_path=scenario_label,
                    scenario_hash=scenario_hash,
                    episode_seed=episode_seed,
                    attempt_idx=attempt_idx,
                    attempt_id=attempt_id,
                ),
            )

        run_trust = TrustScores.from_snapshot(trust_snapshot)
        self._validate_frozen_trust(run_trust.snapshot())
        if (self._trust_schedule is not None
                and run_trust.snapshot() != self._trust_schedule.snapshot(trajectory_idx, episode_idx)):
            raise RuntimeError("Episode input differs from the expected trust schedule")

        agents = self._build_agents(malicious_id, run_trust, scenario)

        episode = Episode(
            scenario,
            agents,
            episode_idx,
            run_trust.snapshot(),
            self.logger,
            self.config,
            run_idx=run_num,
            rng=random.Random(episode_seed),
            run_seed=run_seed,
            trajectory_idx=trajectory_idx,
            trajectory_seed=trajectory_seed,
            episode_seed=episode_seed,
            episode_position=episode_position,
            scenario_idx=scenario_idx,
            scenario_path=scenario_label,
            scenario_hash=scenario_hash,
            attempt_idx=attempt_idx,
            attempt_id=attempt_id,
        )
        saved = None
        if self._is_brs:
            saved = self._brs_store.load(self._brs_store.key(trajectory_idx, episode_idx))
        if saved is None:
            record = episode.run()
        else:
            record = EpisodeRecord.model_validate(saved["record"])
            if (record.attempt_id != attempt_id or record.scenario_hash != scenario_hash
                    or record.scenario_path != scenario_label or record.trust_before != trust_snapshot
                    or record.episode_seed != episode_seed):
                raise StateStoreError("Pending BRS episode conflicts with the checkpoint/scenario plan")

            for agent in agents:
                for name in ("episode_idx", "episode_position", "trajectory_idx", "run_idx",
                             "trajectory_seed", "episode_seed", "scenario_idx", "scenario_path",
                             "scenario_hash", "negotiation_id", "attempt_idx", "attempt_id"):
                    setattr(agent, name, getattr(record, name))
        record = record.model_copy(update={
            "run_idx": run_num,
            "trajectory_idx": trajectory_idx,
            "trajectory_seed": trajectory_seed,
            "episode_idx": episode_idx,
            "episode_position": episode_position,
            "episode_seed": episode_seed,
            "run_seed": episode_seed,
            "scenario_idx": scenario_idx,
            "scenario_path": scenario_label,
            "scenario_hash": scenario_hash,
            "trust_mode": self.config.trust.mode,
            "trust_is_frozen": self._trust_is_frozen,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
            "attempt_idx": attempt_idx,
            "attempt_id": attempt_id,
            **({"trust_replay": self._trust_replay.provenance(trajectory_idx, episode_idx)}
               if self._trust_replay is not None else {}),
            **({"trust_random": self._trust_random.provenance(trajectory_idx, episode_idx)}
               if self._trust_random is not None else {}),
        })

        if self._is_brs:
            record = self._finish_brs_episode(record, scenario, agents, run_trust,
                                              total_rounds, display_names)
        elif not self._trust_is_frozen and self._trust_schedule is None:
            for agent in agents:
                new_focal_trust = agent.calculate_trust_after_episode(
                    episode_idx=episode_idx,
                    history=record.rounds,
                    total_rounds=total_rounds,
                    display_names=display_names,
                )
                run_trust.set_view_for(agent.agent_id, new_focal_trust)
        record = record.model_copy(update={"trust_after": run_trust.snapshot()})
        if self._trust_schedule is not None:
            self._validate_scheduled_trust_record(record.model_dump())

        if (not self._trust_is_frozen and self._trust_schedule is None
                and hasattr(self.logger, "log_oracle_trust")
                and self.config.trust.mode in {"metrics", "freeform"}):
            cover_profiles = {
                aid: {"scores": cfg.scores, "threshold": cfg.min_threshold}
                for aid, cfg in scenario.agent_configs.items()
            }
            for agent_id in scenario.all_agent_ids():
                focal_prev = record.trust_before.get(agent_id, {})
                if not focal_prev:
                    continue
                oracle_scores, metrics_detail = self._oracle.update_with_detail(
                    focal_prev, record, agent_id, cover_profiles=cover_profiles,
                )
                self.logger.log_oracle_trust(
                    episode_idx=episode_idx,
                    run_idx=run_num,
                    focal_agent_id=agent_id,
                    oracle_scores=oracle_scores,
                    metrics_detail=metrics_detail,
                    display_names=display_names,
                    trust_mode=self.config.trust.mode,
                    difficulty=getattr(scenario, "difficulty", ""),
                    **self._identity_fields(
                        trajectory_idx=trajectory_idx,
                        trajectory_seed=trajectory_seed,
                        episode_idx=episode_idx,
                        scenario_idx=scenario_idx,
                        scenario_path=scenario_label,
                        scenario_hash=scenario_hash,
                        episode_seed=episode_seed,
                        attempt_idx=attempt_idx,
                        attempt_id=attempt_id,
                    ),
                )

        if hasattr(self.logger, "log_episode_summary"):
            self.logger.log_episode_summary(
                record,
                display_names=display_names,
                run_idx=run_num,
            )

        if hasattr(self.logger, "log_run_completed"):
            self.logger.log_run_completed(
                episode_idx=episode_idx,
                run_idx=run_num,
                **self._identity_fields(
                    trajectory_idx=trajectory_idx,
                    trajectory_seed=trajectory_seed,
                    episode_idx=episode_idx,
                    scenario_idx=scenario_idx,
                    scenario_path=scenario_label,
                    scenario_hash=scenario_hash,
                    episode_seed=episode_seed,
                    attempt_idx=attempt_idx,
                    attempt_id=attempt_id,
                ),
            )

        return record, run_trust.snapshot()

    def _preflight_brs_feedback_retries(self, progress: dict[int, int]) -> None:
        for key, observer_id in sorted(self._retry_brs_feedback):
            pending = self._brs_store.load(key)
            if pending is None or observer_id not in pending["observers"]:
                raise StateStoreError(f"BRS retry target does not exist: {key}:{observer_id}")

            if pending["observers"][observer_id]["phase"] in {"validated", "fallback_no_update"}:
                continue
            pending = self._brs_store.validate_feedback_retry(key, observer_id)
            record = pending["record"]
            trajectory = record["trajectory_idx"]
            episode = record["episode_idx"]
            if (trajectory not in progress or episode != progress[trajectory] + 1
                    or episode > self.config.game.num_episodes):
                raise StateStoreError(f"BRS retry target is not an unfinished next episode: {key}:{observer_id}")

    @staticmethod
    def _serialize_evidence(state: dict) -> dict:
        return serialize_observer_state(state)

    def _restore_brs_state(self, trajectory_idx: int, raw: Any, trust: dict) -> None:
        ids = self._scenarios[0].all_agent_ids()
        if not isinstance(raw, dict) or set(raw) != set(ids):
            raise StateStoreError("Missing or invalid accumulated BRS evidence")
        state = deserialize_observer_state(raw)
        if trust != scores_from_observer_state(state):
            raise StateStoreError("BRS checkpoint scores do not match accumulated evidence")
        self._brs_state[trajectory_idx] = state

    def _brs_metadata(self) -> dict:
        from src.prompts.feedback_prompt import FEEDBACK_RUBRIC
        from src.BRS.prompt_support import brs_prompt_fingerprint
        from src.BRS.feedback import build_agent_feedback_schema
        ids = self._scenarios[0].all_agent_ids()
        return {
            **self.config.trust.brs.model_dump(),
            "protocol": "brs_agent_feedback_v3",
            "feedback_history_format": "shared_format_history_full_episode",
            "feedback_output": "per_counterpart_scratchpad_and_nonnegative_integer_P_N",
            "feedback_normalization": "r=P/(P+N);s=N/(P+N);zero_counts=(0,0)",
            "prior": {"alpha": 1, "beta": 1},
            "feedback_prompt_sha256": hashlib.sha256(FEEDBACK_RUBRIC.encode()).hexdigest(),
            "prompt_bundle_sha256": brs_prompt_fingerprint(),
            "feedback_schema_sha256": {
                observer: hashlib.sha256(json.dumps(
                    build_agent_feedback_schema([target for target in ids if target != observer]),
                    sort_keys=True,
                ).encode()).hexdigest() for observer in ids
            },
        }

    def _persist_brs_usage(self) -> None:
        self._brs_store.save_usage(self._usage_segment, self._aggregate_usage())

    def _finish_brs_episode(self, record: EpisodeRecord, scenario: Scenario,
                            agents: list, run_trust: TrustScores, total_rounds: int,
                            display_names: dict) -> EpisodeRecord:
        key = self._brs_store.key(record.trajectory_idx, record.episode_idx)
        ids = scenario.all_agent_ids()
        before = self._serialize_evidence(self._brs_state[record.trajectory_idx])
        identity = self._identity_fields(
            trajectory_idx=record.trajectory_idx, trajectory_seed=record.trajectory_seed,
            episode_idx=record.episode_idx, scenario_idx=record.scenario_idx,
            scenario_path=record.scenario_path, scenario_hash=record.scenario_hash,
            episode_seed=record.episode_seed, attempt_idx=record.attempt_idx,
            attempt_id=record.attempt_id,
        )
        identity.update(episode_idx=record.episode_idx, run_idx=record.run_idx,
                        feedback_key=key, difficulty=record.difficulty)
        def observer_context(agent: Any) -> dict:
            return {
                "agent_id": agent.agent_id, "trust_mode": "brs",
                "model_name": agent.model_name,
                "is_malicious": bool(getattr(agent.config, "is_malicious", False)),
                "agent_type": getattr(agent.config, "agent_type", "") or "",
                "client_provenance": self._agent_client_provenance[agent.agent_id],
                "display_names": display_names,
            }
        def feedback_context(saved: dict, observer_id: str) -> dict:
            return {
                "feedback_request_attempt": saved.get("feedback_request_attempt", 1),
                "feedback_correction_attempt": saved.get("feedback_correction_attempt", 0),
                "retry_reason": saved.get("retry_reason"),
                "trust_scores_prompted": saved.get("metadata", {}).get("trust_scores_prompted", {
                    target: (value if self.config.prompts.prompt_version == "v4"
                             else float(f"{value:.2f}"))
                    for target, value in record.trust_before[observer_id].items()
                }),
            }
        pending = self._brs_store.load(key)
        public_history = format_history(record.rounds, len(record.rounds), scenario.parties)
        if pending is None:
            pending = self._brs_store.prepare(
                key, record=record.model_dump(mode="json"),
                public_history=public_history, display_names=display_names, evidence_before=before,
                observer_ids=ids,
            )
        self._persist_brs_usage()
        if pending["evidence_before"] != before:
            raise StateStoreError("Pending BRS update does not follow this trajectory's committed evidence")
        if set(pending["observers"]) != set(ids):
            raise StateStoreError("Pending BRS observer coverage differs from this scenario")
        if pending["public_history"] != public_history or pending["display_names"] != display_names:
            raise StateStoreError("Saved BRS history differs from the shared episode history")
        feedback_by_observer = {}
        skipped_observers = set()
        for agent in agents:
            observer_id = agent.agent_id
            observer = pending["observers"][observer_id]
            targets = [aid for aid in ids if aid != observer_id]
            dispatch_request = None
            if observer["phase"] == "failed":
                pending = self._brs_store.recover_saved_feedback(key, observer_id)
                observer = pending["observers"][observer_id]
            recovered = observer.get("local_validation_recovery")
            if recovered:
                original = recovered["original_failure"]
                self.logger.log_brs_feedback({**identity, **observer_context(agent),
                    "status": "failed", "error": original.get("error"),
                    **feedback_context(original, observer_id),
                    "metadata": original.get("metadata", {})})

            for prior in observer.get("prior_attempts", []):
                metadata = prior.get("metadata", {})
                self.logger.log_brs_feedback({**identity, **observer_context(agent),
                    "status": prior["phase"], "error": prior.get("error"),
                    **feedback_context(prior, observer_id),
                    "metadata": metadata})
                log_saved_call(self.logger, identity=identity, agent_id=observer_id,
                               role="trust_feedback", request=prior.get("request"),
                               metadata=metadata)

            def resolve_invalid_feedback(current: dict) -> tuple[dict, dict | None, bool]:
                saved = current["observers"][observer_id]
                if saved["phase"] != "failed":
                    return current, None, False

                self.logger.log_brs_feedback({**identity, **observer_context(agent),
                    "status": "failed", "error": saved.get("error"),
                    **feedback_context(saved, observer_id),
                    "metadata": saved.get("metadata", {})})
                log_saved_call(self.logger, identity=identity, agent_id=observer_id,
                               role="trust_feedback", request=saved.get("request"),
                               metadata=saved.get("metadata", {}))
                if saved_feedback_is_blocked(saved):
                    return self._brs_store.fallback_feedback(key, observer_id), None, True
                if output_correction_error(saved) is None:
                    correction = agent.build_brs_feedback_correction_request(saved["request"])
                    current = self._brs_store.retry_feedback_output(key, observer_id, correction)
                    return current, correction, True
                candidate = {**saved, "feedback_correction_attempt": 0}
                if (saved.get("feedback_correction_attempt") == 1
                        and output_correction_error(candidate) is None):
                    return self._brs_store.fallback_feedback(key, observer_id), None, True
                return current, None, False

            pending, dispatch_request, resolved = resolve_invalid_feedback(pending)
            observer = pending["observers"][observer_id]
            if observer["phase"] in {"requested", "failed"}:
                metadata = observer.get("metadata", {})
                self.logger.log_brs_feedback({**identity, **observer_context(agent),
                    "status": observer["phase"], "error": observer.get("error"),
                    **feedback_context(observer, observer_id),
                    "metadata": metadata})
                log_saved_call(self.logger, identity=identity, agent_id=observer_id,
                               role="trust_feedback", request=observer.get("request"),
                               metadata=metadata)
                retry_target = (key, observer_id)
                if dispatch_request is not None:
                    pass
                elif (retry_target in self._retry_brs_feedback
                        and retry_target not in self._consumed_brs_retries):
                    pending = self._brs_store.retry_feedback(key, observer_id)
                    self._consumed_brs_retries.add(retry_target)
                    observer = pending["observers"][observer_id]
                    dispatch_request = observer["request"]
                else:
                    reason = transport_retry_error(observer)
                    recovery_hint = (
                        f" Retry only this transiently failed request with --resume "
                        f"--retry-brs-feedback {key}:{observer_id}."
                        if reason is None else f" Explicit feedback retry is unavailable: {reason}."
                    )
                    raise FeedbackError(
                        f"BRS feedback for {observer_id} was already dispatched without a saved valid "
                        f"response ({key}). Original error: {observer.get('error', 'request outcome unknown')}."
                        + recovery_hint
                    )
            if observer["phase"] == "prepared":
                request = agent.build_brs_feedback_request(record.rounds, total_rounds, display_names)
                pending = self._brs_store.mark_requested(key, observer_id, request)
                observer = pending["observers"][observer_id]
                dispatch_request = request
            while dispatch_request is not None:
                request = dispatch_request
                request_context = feedback_context(observer, observer_id)
                self.logger.log_brs_feedback({**identity, **observer_context(agent),
                    "status": "requested", **request_context})
                try:
                    assessment = agent.provide_brs_feedback(request)
                    counts = assessment.feedback
                    feedback = normalize_counts(counts)
                except Exception as exc:
                    metadata = dict(getattr(agent, "brs_feedback_metadata", {}) or {})
                    metadata.update(request_context)
                    pending = self._brs_store.fail_feedback(key, observer_id, str(exc), metadata=metadata)
                    self.logger.log_brs_feedback({**identity, **observer_context(agent),
                        "status": "failed", "error": str(exc),
                        **request_context, "metadata": metadata})
                    log_saved_call(self.logger, identity=identity, agent_id=observer_id,
                                   role="trust_feedback", request=request, metadata=metadata)
                    pending, dispatch_request, resolved = resolve_invalid_feedback(pending)
                    observer = pending["observers"][observer_id]
                    if not resolved:
                        raise
                    if dispatch_request is not None:
                        continue
                    break
                else:
                    metadata = dict(getattr(agent, "brs_feedback_metadata", {}) or {})
                    metadata.update(request_context)
                    pending = self._brs_store.save_feedback(
                        key, observer_id,
                        feedback={aid: pair.model_dump() for aid, pair in feedback.items()},
                        counts={aid: value.model_dump() for aid, value in counts.items()},
                        scratchpad=dict(assessment.scratchpad),
                        metadata=metadata,
                        usage_snapshot={"segment": self._usage_segment,
                                        "negotiator": self._aggregate_usage()},
                    )
                finally:
                    self._persist_brs_usage()
                observer = pending["observers"][observer_id]
                dispatch_request = None
            observer = pending["observers"][observer_id]
            if observer["phase"] == "fallback_no_update":
                failed = {**observer, "phase": "failed"}
                candidate = {**failed, "feedback_correction_attempt": 0}
                if (observer.get("update_skipped") is not True
                        or any(observer.get(field) is not None for field in ("counts", "feedback", "scratchpad"))
                        or not (saved_feedback_is_blocked(failed)
                                or observer.get("feedback_correction_attempt") == 1
                                and output_correction_error(candidate) is None)):
                    raise StateStoreError("Saved BRS fallback lacks an eligible recorded output failure")
                skipped_observers.add(observer_id)
                self.logger.log_brs_feedback({**identity, **observer_context(agent),
                    "status": "fallback_no_update", "counts": None, "feedback": None,
                    "scratchpad": None, "update_skipped": True,
                    "fallback_reason": observer.get("fallback_reason"),
                    **feedback_context(observer, observer_id),
                    "metadata": observer.get("metadata", {})})
                log_saved_call(self.logger, identity=identity, agent_id=observer_id,
                               role="trust_feedback", request=observer.get("request"),
                               metadata=observer.get("metadata", {}))
                continue
            assessment = parse_agent_feedback_output(json.dumps({
                "scratchpad": observer.get("scratchpad"), "feedback": observer.get("counts"),
            }), targets)
            counts = assessment.feedback
            feedback = parse_feedback_output(json.dumps(observer.get("feedback")), targets)
            if feedback != normalize_counts(counts):
                raise StateStoreError("Saved BRS feedback does not match normalized agent counts")
            feedback_by_observer[observer_id] = feedback

            self.logger.log_brs_feedback({**identity, **observer_context(agent),
                "status": "validated", "counts": observer["counts"],
                **feedback_context(observer, observer_id),
                "scratchpad": dict(assessment.scratchpad),
                "feedback": observer["feedback"], "metadata": observer.get("metadata", {})})
            log_saved_call(self.logger, identity=identity, agent_id=observer_id,
                           role="trust_feedback", request=observer.get("request"),
                           metadata=observer.get("metadata", {}), scratchpad=assessment.scratchpad)
        restored_before = deserialize_observer_state(before)
        after = {
            observer: (restored_before[observer]
                       if observer in skipped_observers else
                       update_brs(before[observer], feedback_by_observer[observer],
                                  self.config.trust.brs.forgetting_factor))
            for observer in ids
        }
        serialized_after = self._serialize_evidence(after)
        scores = scores_from_observer_state(after)
        if pending["phase"] == "feedback_ready":
            pending = self._brs_store.stage(key, evidence_after=serialized_after, scores=scores)
        elif pending["evidence_after"] != serialized_after or pending["scores"] != scores:
            raise StateStoreError("Saved BRS evidence/scores do not match the saved feedback")
        for agent in agents:
            filtered = counterpart_scores(scores, agent.agent_id)
            if agent.agent_id not in pending["delivered"]:
                try:
                    result = agent.calculate_trust_after_episode(
                        episode_idx=record.episode_idx, history=record.rounds,
                        total_rounds=total_rounds, display_names=display_names,
                        brs_scores=filtered,
                    )
                    if result != filtered:
                        raise StateStoreError("Negotiator altered authoritative BRS scores")
                except Exception as exc:
                    self.logger.log_brs_tool({**identity, **observer_context(agent),
                        "status": "failed", "error": str(exc),
                        "metadata": getattr(agent, "brs_tool_metadata", {})})
                    log_saved_call(self.logger, identity=identity, agent_id=agent.agent_id,
                                   role="trust_tool", metadata=getattr(agent, "brs_tool_metadata", {}))
                    raise
                finally:
                    self._persist_brs_usage()
                pending = self._brs_store.mark_delivered(
                    key, agent.agent_id, metadata=getattr(agent, "brs_tool_metadata", {})
                )
            else:
                agent.trust_scores = dict(filtered)
            self.logger.log_brs_tool({**identity, **observer_context(agent),
                "status": "delivered", "scores": filtered,
                "metadata": pending.get("delivery_metadata", {}).get(agent.agent_id, {})})
            log_saved_call(self.logger, identity=identity, agent_id=agent.agent_id,
                           role="trust_tool",
                           metadata=pending.get("delivery_metadata", {}).get(agent.agent_id, {}))
            run_trust.set_view_for(agent.agent_id, filtered)

        for agent in agents:
            observer_id = agent.agent_id
            observer = pending["observers"][observer_id]
            scratchpad = observer["scratchpad"] or {}
            self.logger.log_trust_update(
                **identity, **observer_context(agent),
                scratchpad="\n".join(f"{target}: {value}" for target, value in scratchpad.items()),
                scratchpad_by_agent=dict(scratchpad),
                trust_scores_before=dict(record.trust_before[observer_id]),
                trust_scores_prompted=observer.get("metadata", {}).get("trust_scores_prompted"),
                trust_scores_after=dict(scores[observer_id]),
                brs_counts=observer["counts"], brs_feedback=observer["feedback"],
                brs_feedback_status=observer["phase"],
                brs_update_skipped=observer["phase"] == "fallback_no_update",
                brs_fallback_reason=observer.get("fallback_reason"),
                brs_state_before=before[observer_id], brs_state_after=serialized_after[observer_id],
                brs_forgetting_factor=self.config.trust.brs.forgetting_factor,
                trust_update_applied_by="runner", trust_update_protocol="brs_agent_feedback_v3",
            )
        return record.model_copy(update={"brs_state_before": before,
            "brs_state_after": serialized_after, "brs_feedback_key": key})

    def _select_malicious_agent(self, episode_idx: int) -> str:
        del episode_idx
        if not self.config.malicious.enabled:
            raise ValueError("This implementation expects malicious.enabled to be true")
        if self.config.malicious.agent_id is not None:
            return self.config.malicious.agent_id

        return self._scenarios[0].all_agent_ids()[-1]

    def _build_agents(self, malicious_id: str, trust: TrustScores, scenario: Scenario) -> list[Any]:
        agents = []
        for agent_id in scenario.all_agent_ids():
            base_config = scenario.agent_configs[agent_id]
            is_malicious = agent_id == malicious_id
            if is_malicious:
                agent_config = base_config.model_copy(
                    update={
                        "is_malicious": True,
                        "agent_type": "malicious_strategic",
                        "min_threshold": self.config.malicious.threshold,
                    }
                )
                agent_cls = MaliciousAgent
            else:
                agent_config = base_config.model_copy(update={"is_malicious": False})
                agent_cls = HonestAgent

            use_trust_scores = self.config.trust.mode != "none"
            agents.append(
                agent_cls(
                    config=agent_config,
                    scenario=scenario,
                    llm_client=self._agent_clients[agent_id],
                    use_trust_scores=use_trust_scores,
                    trust_scores=trust.view_for(agent_id),
                    game_config=self.config,
                    logger=self.logger,
                    model_name=self._agent_model_names.get(agent_id, ""),
                )
            )
        return agents

    @staticmethod
    def _describe_client(client: Any, *, configured_model: str) -> dict[str, Any]:
        endpoint = getattr(client, "base_url", None)
        endpoint_fingerprint = (
            hashlib.sha256(str(endpoint).encode("utf-8")).hexdigest()[:16]
            if endpoint else None
        )
        provenance = {
            "client_class": f"{type(client).__module__}.{type(client).__name__}",
            "is_stub": isinstance(client, StubLLMClient),
            "configured_model": configured_model,
            "resolved_model": getattr(client, "model", None) or configured_model,
            "api_version": getattr(client, "api_version", None),
            "endpoint_fingerprint": endpoint_fingerprint,
            "reasoning_prefixes": list(getattr(client, "reasoning_prefixes", ()) or ()),
            "thinking_budget": getattr(client, "thinking_budget", None),
            "supports_temperature": getattr(client, "_temp_for_reasoning", None),
            "supports_seed": getattr(client, "_seed_supported", None),
        }
        reasoning_effort = getattr(client, "reasoning_effort", None)
        if reasoning_effort is not None:
            provenance["reasoning_effort"] = reasoning_effort
        capability_overrides = {
            "is_reasoning_model": getattr(
                client, "_is_reasoning_model_override", None
            ),
            "merge_system_into_user": getattr(
                client, "_merge_system_into_user_override", None
            ),
            "supports_response_format": getattr(
                client, "_supports_response_format_override", None
            ),
        }
        configured_capabilities = {
            name: value
            for name, value in capability_overrides.items()
            if value is not None
        }
        if configured_capabilities:
            provenance["model_capability_protocol"] = "explicit_v1"
            provenance.update(configured_capabilities)
        openai_response_format_protocol = getattr(
            client, "openai_response_format_protocol", None
        )
        if openai_response_format_protocol is not None:
            provenance["openai_response_format_protocol"] = (
                openai_response_format_protocol
            )
        structured_protocol = getattr(client, "structured_output_protocol", None)
        if structured_protocol is not None:
            provider = getattr(client, "provider", None)
            provenance.update({
                "structured_output_protocol": structured_protocol,
                "provider": provider,
                "backend": getattr(client, "backend", None),
                "structured_output_reformat_attempts": int(
                    getattr(client, "structured_output_reformat_attempts", 0)
                ),
                "structured_output_regeneration_attempts": int(
                    getattr(
                        client,
                        "structured_output_regeneration_attempts",
                        0,
                    )
                ),
                "trajectory_cell_retry_attempts": int(
                    getattr(client, "trajectory_cell_retry_attempts", 0)
                ),
                "scratchpad_limits": dict(
                    getattr(client, "scratchpad_limits", {}) or {}
                ),
                "structured_output_recovery_protocol": getattr(
                    client,
                    "structured_output_recovery_protocol",
                    None,
                ),
                "structured_output_regeneration_protocol": getattr(
                    client,
                    "structured_output_regeneration_protocol",
                    None,
                ),
                "trajectory_cell_retry_protocol": getattr(
                    client,
                    "trajectory_cell_retry_protocol",
                    None,
                ),
                "scratchpad_recovery_targets": {
                    str(key): list(values)
                    for key, values in dict(
                        getattr(client, "scratchpad_recovery_targets", {}) or {}
                    ).items()
                },
                "provider_sdk": (
                    "google-genai" if provider == "gemini" else "openai"
                ),
                "provider_sdk_version": (
                    getattr(client, "google_genai_version", None)
                    if provider == "gemini"
                    else getattr(client, "openai_sdk_version", None)
                ),
            })
            if provider in {"kimi", "deepseek"}:
                provenance.update({
                    "thinking_mode": getattr(
                        client, "thinking_mode", None
                    ),
                    "thinking_keep": getattr(
                        client, "thinking_keep", None
                    ),
                    "max_token_parameter": getattr(
                        client, "max_token_parameter", None
                    ),
                    "require_seed_support": getattr(
                        client, "_require_seed_support", False
                    ),
                })
        return provenance

    def _execution_backend(self) -> str:
        stub_flags = {
            bool(details.get("is_stub"))
            for details in self._agent_client_provenance.values()
        }
        if stub_flags == {True}:
            return "stub"
        if stub_flags == {False}:
            return "provider"
        return "mixed"

    def _trajectory_count(self) -> int:
        count = getattr(self.config.game, "num_trajectories", None)
        legacy = getattr(self.config.game, "num_runs", None)
        if count is None:
            count = legacy if legacy is not None else 1
        if legacy is not None and legacy != count:
            raise ValueError(
                "game.num_runs and game.num_trajectories disagree; update both "
                "or remove the deprecated num_runs field"
            )
        if count < 1:
            raise ValueError("game.num_trajectories must be at least 1")
        return int(count)

    def _scenario_index(
        self,
        trajectory_idx: int,
        episode_position: int | None = None,
        *,
        episode_idx: int | None = None,
    ) -> int:
        episode_ids = [
            int(value)
            for value in (episode_position, episode_idx)
            if value is not None
        ]
        if not episode_ids:
            raise ValueError("An episode identifier is required")
        if len(set(episode_ids)) != 1:
            raise ValueError("episode_idx must equal episode_position")
        episode_position = episode_ids[0]

        n_scenarios = len(self._scenarios)
        policy = getattr(self.config.game, "scenario_order_policy", "fixed")
        if policy == "fixed":
            offset = 0
        elif policy == "cyclic_counterbalanced":
            offset = trajectory_idx - 1
        else:
            raise ValueError(f"Unsupported scenario_order_policy {policy!r}")
        return ((episode_position - 1 + offset) % n_scenarios) + 1

    def _scenario_orders(
        self,
        total_positions: int | None = None,
        *,
        total_episodes: int | None = None,
    ) -> dict[str, list[int]]:
        if (
            total_positions is not None
            and total_episodes is not None
            and total_positions != total_episodes
        ):
            raise ValueError("total_positions must equal total_episodes")
        episodes = total_positions or total_episodes or self.config.game.num_episodes
        return {
            str(trajectory_idx): [
                self._scenario_index(trajectory_idx, episode_idx)
                for episode_idx in range(1, episodes + 1)
            ]
            for trajectory_idx in range(1, self._trajectory_count() + 1)
        }

    def _validate_seed_plan(self) -> None:
        trajectory_seeds = {
            trajectory_idx: derive_trajectory_seed(self._base_seed, trajectory_idx)
            for trajectory_idx in range(1, self._trajectory_count() + 1)
        }
        if len(set(trajectory_seeds.values())) != len(trajectory_seeds):
            raise RuntimeError(
                "Trajectory seed collision detected; choose a different base_seed"
            )

        episode_seeds: set[int] = set()
        for trajectory_idx, trajectory_seed in trajectory_seeds.items():
            for episode_idx in range(1, self.config.game.num_episodes + 1):
                scenario_idx = self._scenario_index(
                    trajectory_idx, episode_idx
                )
                scenario_path = self._scenario_paths[scenario_idx - 1]
                scenario_hash = self._scenario_hashes[scenario_idx - 1]
                seed = derive_episode_seed(
                    trajectory_seed,
                    episode_idx,
                    self._scenario_identity(
                        scenario_idx, scenario_path, scenario_hash
                    ),
                )
                if seed in episode_seeds:
                    raise RuntimeError(
                        "Episode seed collision detected; choose a different base_seed"
                    )
                episode_seeds.add(seed)

    @staticmethod
    def _scenario_identity(
        scenario_idx: int,
        scenario_label: str,
        scenario_hash: str,
    ) -> str:
        return f"{scenario_idx}:{scenario_hash or scenario_label or 'in-memory'}"

    @staticmethod
    def _identity_fields(
        *,
        trajectory_idx: int,
        trajectory_seed: int,
        episode_position: int | None = None,
        episode_idx: int | None = None,
        scenario_idx: int,
        scenario_path: str,
        scenario_hash: str,
        episode_seed: int,
        attempt_idx: int,
        attempt_id: str,
    ) -> dict[str, Any]:
        episode_ids = [
            int(value)
            for value in (episode_position, episode_idx)
            if value is not None
        ]
        if not episode_ids:
            raise ValueError("An episode identifier is required")
        if len(set(episode_ids)) != 1:
            raise ValueError("episode_idx must equal episode_position")
        episode_position = episode_ids[0]
        negotiation_id = f"t{trajectory_idx:03d}-e{episode_position:03d}"
        return {
            "trajectory_idx": trajectory_idx,
            "trajectory_seed": trajectory_seed,
            "episode_position": episode_position,
            "scenario_idx": scenario_idx,
            "scenario_path": scenario_path,
            "scenario_hash": scenario_hash,
            "episode_seed": episode_seed,
            "negotiation_id": negotiation_id,
            "attempt_idx": attempt_idx,
            "attempt_id": attempt_id,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
        }

    def _collect_trajectory_results(
        self,
        future_items: list[tuple[int, dict[str, Any], Any]],
    ) -> tuple[
        dict[int, tuple[EpisodeRecord, dict[str, dict[str, float]]]],
        dict[int, tuple[dict[str, Any], Exception, str]],
    ]:
        results: dict[int, tuple[EpisodeRecord, dict[str, dict[str, float]]]] = {}
        failed: dict[int, tuple[dict[str, Any], Exception, str]] = {}
        for trajectory_idx, item, future in future_items:
            try:
                results[trajectory_idx] = future.result()
            except Exception as exc:
                failed[trajectory_idx] = (item, exc, traceback.format_exc())
        return results, failed

    def _execute_trajectory_batch(
        self,
        work_items: list[dict[str, Any]],
        num_workers: int,
    ) -> tuple[
        dict[int, tuple[EpisodeRecord, dict[str, dict[str, float]]]],
        dict[int, tuple[dict[str, Any], Exception, str]],
    ]:
        if num_workers > 1 and len(work_items) > 1:
            with ThreadPoolExecutor(
                max_workers=min(num_workers, len(work_items))
            ) as pool:
                future_items = [
                    (
                        int(item["trajectory_idx"]),
                        item,
                        pool.submit(self._run_single, **item),
                    )
                    for item in work_items
                ]
                return self._collect_trajectory_results(future_items)

        results: dict[
            int,
            tuple[EpisodeRecord, dict[str, dict[str, float]]],
        ] = {}
        failures: dict[int, tuple[dict[str, Any], Exception, str]] = {}
        for item in work_items:
            trajectory_idx = int(item["trajectory_idx"])
            try:
                results[trajectory_idx] = self._run_single(**item)
            except Exception as exc:
                failures[trajectory_idx] = (item, exc, traceback.format_exc())
        return results, failures

    def _gemini_cell_retry_attempts(self) -> int:
        return max(
            (
                int(getattr(client, "trajectory_cell_retry_attempts", 0) or 0)
                for client in self._agent_clients.values()
                if bool(
                    getattr(client, "native_structured_output_enabled", False)
                )
            ),
            default=0,
        )

    @staticmethod
    def _next_cell_attempt(
        item: dict[str, Any],
        attempt_counts: dict[int, dict[int, int]],
    ) -> dict[str, Any]:
        trajectory_idx = int(item["trajectory_idx"])
        episode_idx = int(item["episode_position"])
        attempt_idx = attempt_counts[trajectory_idx].get(episode_idx, 0) + 1
        attempt_counts[trajectory_idx][episode_idx] = attempt_idx
        retry_item = dict(item)
        retry_item["attempt_idx"] = attempt_idx
        retry_item["attempt_id"] = (
            f"t{trajectory_idx:03d}-e{episode_idx:03d}-a{attempt_idx:03d}"
        )
        return retry_item

    def _log_run_failure(
        self,
        item: dict[str, Any],
        exc: Exception,
        *,
        traceback_text: str | None = None,
        retry_disposition: str = "fatal",
        cell_retry_attempts: int = 0,
    ) -> None:
        episode_idx = int(
            item.get("episode_position", item.get("episode_idx", item.get("ep_num", 0)))
        )
        trajectory_idx = int(item["trajectory_idx"])
        print(
            f"  Warning: episode {episode_idx} trajectory {trajectory_idx} failed: {exc}",
            file=sys.stderr,
        )
        if hasattr(self.logger, "log_run_failure"):
            self.logger.log_run_failure(
                episode_idx=episode_idx,
                run_idx=trajectory_idx,
                error=str(exc),
                traceback=traceback_text or traceback.format_exc(),
                retry_disposition=retry_disposition,
                cell_retry_attempts=cell_retry_attempts,
                **self._identity_fields(
                    trajectory_idx=trajectory_idx,
                    trajectory_seed=int(item["trajectory_seed"]),
                    episode_position=episode_idx,
                    scenario_idx=int(item["scenario_idx"]),
                    scenario_path=str(item["scenario_label"]),
                    scenario_hash=str(item["scenario_hash"]),
                    episode_seed=int(item["episode_seed"]),
                    attempt_idx=int(item["attempt_idx"]),
                    attempt_id=str(item["attempt_id"]),
                ),
            )

    def _write_checkpoint(
        self,
        progress_by_trajectory: dict[int, int],
        trust_by_trajectory: dict[int, TrustScores],
        attempt_counts: dict[int, dict[int, int]],
        failed_trajectories: set[int] | None = None,
        retrying_trajectories: set[int] | None = None,
        exhausted_trajectories: set[int] | None = None,
    ) -> None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return
        if self._is_brs:
            self._persist_brs_usage()
        failed = failed_trajectories or set()
        retrying = retrying_trajectories or set()
        exhausted = exhausted_trajectories or set()
        checkpoint_path = Path(run_dir) / "checkpoint.json"
        tmp_path = Path(run_dir) / "checkpoint.json.tmp"
        committed_episode = min(progress_by_trajectory.values(), default=0)
        scenario_orders = self._scenario_orders()
        data: dict[str, Any] = {
            "schema_version": 4 if self._is_brs else 2,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
            "seed_scheme_version": SEED_SCHEME_VERSION,
            "last_committed_position": committed_episode,
            "last_completed_episode": committed_episode,
            "num_episodes": self.config.game.num_episodes,
            "num_trajectories": self._trajectory_count(),
            "scenario_order_policy": getattr(
                self.config.game, "scenario_order_policy", "fixed"
            ),
            "scenario_orders": scenario_orders,
            "scenario_plan_fingerprint": self._scenario_plan_fingerprint(
                scenario_orders
            ),
            "trajectories": {
                str(trajectory_idx): {
                    "trajectory_idx": trajectory_idx,
                    "trajectory_seed": derive_trajectory_seed(
                        self._base_seed, trajectory_idx
                    ),
                    "last_completed_episode": progress_by_trajectory[trajectory_idx],
                    "trust": trust_by_trajectory[trajectory_idx].snapshot(),
                    "attempt_counts": {
                        str(episode_idx): count
                        for episode_idx, count in sorted(
                            attempt_counts.get(trajectory_idx, {}).items()
                        )
                    },
                    "status": (
                        "retry_pending"
                        if trajectory_idx in retrying
                        else "retry_exhausted"
                        if trajectory_idx in exhausted
                        else "failed"
                        if trajectory_idx in failed
                        else "complete"
                        if progress_by_trajectory[trajectory_idx]
                        >= self.config.game.num_episodes
                        else "active"
                    ),
                }
                for trajectory_idx in sorted(progress_by_trajectory)
            },
            "llm_usage_current_segment": self._aggregate_usage(),
            "llm_usage_cumulative": self._cumulative_usage(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_config_fingerprint": self._build_config_fingerprint(),
        }
        if self._is_brs:
            for idx, row in data["trajectories"].items():
                row["brs_state"] = self._serialize_evidence(self._brs_state[int(idx)])
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, checkpoint_path)
            if self._is_brs:
                directory = os.open(Path(run_dir), os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                for idx, progress in progress_by_trajectory.items():
                    if progress:
                        key = self._brs_store.key(idx, progress)
                        saved = self._brs_store.load(key)
                        if saved and saved["phase"] == "delivered":
                            self._brs_store.mark_committed(key)
        except OSError as write_exc:
            if self._is_brs:
                raise StateStoreError("Could not durably checkpoint BRS state") from write_exc
            print(
                f"Warning: could not write checkpoint.json: {write_exc}",
                file=sys.stderr,
            )

    def _write_run_status(
        self,
        *,
        status: str,
        progress_by_trajectory: dict[int, int],
        attempt_counts: dict[int, dict[int, int]],
        incomplete_trajectories: set[int],
    ) -> None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return
        root = Path(run_dir)
        path = root / "run_status.json"
        temporary = root / "run_status.json.tmp"
        payload = {
            "status": status,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
            "num_episodes": self.config.game.num_episodes,
            "num_trajectories": self._trajectory_count(),
            "completed_trajectories": sorted(
                trajectory_idx
                for trajectory_idx, progress in progress_by_trajectory.items()
                if progress >= self.config.game.num_episodes
            ),
            "incomplete_trajectories": sorted(incomplete_trajectories),
            "progress_by_trajectory": {
                str(key): value
                for key, value in sorted(progress_by_trajectory.items())
            },
            "attempt_counts": {
                str(trajectory_idx): {
                    str(episode_idx): count
                    for episode_idx, count in sorted(counts.items())
                }
                for trajectory_idx, counts in sorted(attempt_counts.items())
            },
            "trajectory_cell_retry_attempts": (
                self._gemini_cell_retry_attempts()
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as write_exc:
            print(
                f"Warning: could not write run_status.json: {write_exc}",
                file=sys.stderr,
            )

    def _invalidate_incomplete_aggregates(
        self,
        incomplete_trajectories: set[int],
    ) -> None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return
        root = Path(run_dir)
        marker = {
            "schema_version": 2,
            "experiment_status": "incomplete",
            "analysis_available": False,
            "incomplete_trajectories": sorted(incomplete_trajectories),
            "message": (
                "The longitudinal grid is incomplete. Resume the checkpoint "
                "before generating or interpreting aggregate analyses."
            ),
            "run_status_file": "run_status.json",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            experiment_tmp = root / "experiment.json.tmp"
            experiment_tmp.write_text(
                json.dumps(marker, indent=2),
                encoding="utf-8",
            )
            os.replace(experiment_tmp, root / "experiment.json")
            for path in (
                root / "summary.csv",
                root / "trajectory_summary.csv",
                root / "scenario_summary.csv",
            ):
                if path.exists():
                    path.write_text("status\nincomplete\n", encoding="utf-8")
            trust_summary_path = root / "trust_summary.jsonl"
            if trust_summary_path.exists():
                trust_summary_path.write_text("", encoding="utf-8")

            replay_dir = root / "Replay"
            if replay_dir.exists():
                notice = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>Run incomplete</title></head><body>"
                    "<h1>Run incomplete</h1><p>Aggregate analysis is disabled "
                    "until all trajectory episodes are complete. Resume this "
                    "run using its checkpoint.</p></body></html>"
                )
                for html_path in replay_dir.glob("*.html"):
                    html_path.write_text(notice, encoding="utf-8")
        except OSError as write_exc:
            print(
                "Warning: could not invalidate stale aggregate artifacts: "
                f"{write_exc}",
                file=sys.stderr,
            )

    def _load_checkpoint(self) -> tuple[
        dict[int, int],
        dict[int, TrustScores],
        dict[int, dict[int, int]],
        dict[str, Any],
    ] | None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return None
        checkpoint_path = Path(run_dir) / "checkpoint.json"
        if not checkpoint_path.exists():
            return None
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read checkpoint.json safely: {exc}"
            ) from exc

        if data.get("schema_version") != (4 if self._is_brs else 2):
            if self._is_brs:
                raise StateStoreError("BRS resume requires a schema v4 checkpoint with observer-specific evidence")
            raise RuntimeError(
                "Legacy checkpoint detected. It contains one averaged canonical "
                "trust state and cannot be resumed as independent trajectories. "
                "Start the longitudinal design in a new output directory."
            )
        if data.get("design_version") != LONGITUDINAL_DESIGN_VERSION:
            raise RuntimeError(
                "Checkpoint design_version is incompatible with this engine: "
                f"{data.get('design_version')!r}"
            )
        if data.get("seed_scheme_version") != SEED_SCHEME_VERSION:
            raise RuntimeError(
                "Checkpoint seed scheme is incompatible with this engine: "
                f"{data.get('seed_scheme_version')!r}"
            )

        expected_ids = set(range(1, self._trajectory_count() + 1))
        raw_trajectories = data.get("trajectories")
        if not isinstance(raw_trajectories, dict):
            raise RuntimeError("Checkpoint v2 is missing the trajectories mapping")
        try:
            actual_ids = {int(key) for key in raw_trajectories}
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Checkpoint has invalid trajectory identifiers") from exc
        if actual_ids != expected_ids:
            raise RuntimeError(
                "Checkpoint trajectory set does not match the configured "
                f"num_trajectories: checkpoint={sorted(actual_ids)}, "
                f"current={sorted(expected_ids)}"
            )

        progress: dict[int, int] = {}
        trust: dict[int, TrustScores] = {}
        attempts: dict[int, dict[int, int]] = {}
        for trajectory_idx in sorted(expected_ids):
            raw = raw_trajectories[str(trajectory_idx)]
            episode = int(raw.get("last_completed_episode", 0))
            if episode < 0 or episode > self.config.game.num_episodes:
                raise RuntimeError(
                    f"Checkpoint trajectory {trajectory_idx} has invalid progress {episode}"
                )
            trust_snapshot = raw.get("trust")
            if not isinstance(trust_snapshot, dict) or not trust_snapshot:
                raise RuntimeError(
                    f"Checkpoint trajectory {trajectory_idx} has no trust state"
                )
            if (self._trust_schedule is not None
                    and trust_snapshot != self._trust_schedule.snapshot(trajectory_idx, max(1, episode))):
                raise RuntimeError(
                    f"Checkpoint trust differs from expected schedule for trajectory {trajectory_idx}"
                )
            raw_attempts = raw.get("attempt_counts", {})
            if not isinstance(raw_attempts, dict):
                raise RuntimeError(
                    f"Checkpoint trajectory {trajectory_idx} has invalid attempt counts"
                )
            progress[trajectory_idx] = episode
            trust[trajectory_idx] = TrustScores.from_snapshot(trust_snapshot)
            if self._is_brs:
                self._restore_brs_state(trajectory_idx, raw.get("brs_state"), trust_snapshot)
            attempts[trajectory_idx] = {
                int(ep): int(count) for ep, count in raw_attempts.items()
            }
        return progress, trust, attempts, data

    def _load_prior_records(
        self,
        progress_by_trajectory: dict[int, int],
    ) -> list[Any]:
        episodes_path = getattr(self.logger, "episodes_path", None)
        if episodes_path is None:
            if self._trust_schedule is not None and any(progress_by_trajectory.values()):
                raise RuntimeError("Scheduled trust checkpoint requires completed episode logs")
            return []
        path = Path(episodes_path)
        if not path.exists():
            if self._trust_schedule is not None and any(progress_by_trajectory.values()):
                raise RuntimeError("Scheduled trust checkpoint requires completed episode logs")
            return []
        completed_attempts: set[str] | None = None
        incidents_path = getattr(self.logger, "incidents_path", None)
        if (self._trust_schedule is not None and any(progress_by_trajectory.values())
                and (incidents_path is None or not Path(incidents_path).is_file())):
            raise RuntimeError("Scheduled trust checkpoint requires completion incident logs")
        if incidents_path is not None and Path(incidents_path).exists():
            completed_attempts = set()
            try:
                with Path(incidents_path).open(encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            incident = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if incident.get("type") == "run_completed":
                            attempt_id = str(incident.get("attempt_id", ""))
                            if attempt_id:
                                completed_attempts.add(attempt_id)
            except OSError as exc:
                if self._trust_schedule is not None:
                    raise RuntimeError("Cannot validate scheduled trust completion incident logs") from exc
                print(
                    f"Warning: could not filter completed resume attempts: {exc}",
                    file=sys.stderr,
                )
                completed_attempts = None
        latest: dict[tuple[int, int], tuple[int, dict[str, Any]]] = {}
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if completed_attempts is not None:
                        attempt_id = str(row.get("attempt_id", ""))
                        if not attempt_id or attempt_id not in completed_attempts:
                            continue
                    trajectory_idx = int(row.get("trajectory_idx", row.get("run_idx", 1)))
                    episode_idx = int(
                        row.get("episode_position", row.get("episode_idx", 0))
                    )
                    if trajectory_idx not in progress_by_trajectory:
                        continue
                    if episode_idx > progress_by_trajectory[trajectory_idx]:
                        continue
                    if self._trust_schedule is not None:
                        self._validate_scheduled_trust_record(row)
                    key = (trajectory_idx, episode_idx)
                    attempt_idx = int(row.get("attempt_idx", 1))
                    previous = latest.get(key)
                    if previous is None or attempt_idx >= previous[0]:
                        latest[key] = (attempt_idx, row)
        except OSError as exc:
            print(
                f"Warning: could not read episodes.jsonl for resume: {exc}",
                file=sys.stderr,
            )
        if self._trust_schedule is not None:
            expected_keys = {
                (trajectory_idx, episode_idx)
                for trajectory_idx, progress in progress_by_trajectory.items()
                for episode_idx in range(1, progress + 1)
            }
            if set(latest) != expected_keys:
                raise RuntimeError("Scheduled trust checkpoint is missing completed episode logs")
        return [
            _EpisodeSummaryProxy(latest[key][1])
            for key in sorted(latest, key=lambda item: (item[1], item[0]))
        ]

    @staticmethod
    def _scenario_plan_fingerprint(scenario_orders: dict[str, list[int]]) -> str:
        encoded = json.dumps(
            scenario_orders, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _validate_checkpoint_scenario_plan(self, data: dict[str, Any]) -> None:
        stored = data.get("scenario_orders")
        if not isinstance(stored, dict):
            raise RuntimeError("Checkpoint v2 is missing its ordered scenario plan")
        stored_fingerprint = data.get("scenario_plan_fingerprint", "")
        if stored_fingerprint != self._scenario_plan_fingerprint(stored):
            raise RuntimeError("Checkpoint scenario plan fingerprint is invalid")

        current = self._scenario_orders()
        for trajectory_key, stored_order in stored.items():
            current_order = current.get(str(trajectory_key))
            if current_order is None or current_order[:len(stored_order)] != stored_order:
                raise RuntimeError(
                    "Checkpoint scenario order is not a prefix of the current "
                    f"plan for trajectory {trajectory_key}"
                )

    def _reconcile_checkpoint_with_episode_log(
        self,
        progress_by_trajectory: dict[int, int],
        trust_by_trajectory: dict[int, TrustScores],
        attempt_counts: dict[int, dict[int, int]],
    ) -> None:
        episodes_path = getattr(self.logger, "episodes_path", None)
        incidents_path = getattr(self.logger, "incidents_path", None)
        if episodes_path is None or incidents_path is None:
            return
        episodes_file = Path(episodes_path)
        incidents_file = Path(incidents_path)
        if not episodes_file.exists() or not incidents_file.exists():
            return

        completed_attempts: set[str] = set()
        try:
            with incidents_file.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") != "run_completed":
                        continue
                    attempt_id = str(row.get("attempt_id", ""))
                    if attempt_id:
                        completed_attempts.add(attempt_id)
        except OSError as exc:
            print(f"Warning: could not reconcile incidents log: {exc}", file=sys.stderr)
            return

        latest: dict[tuple[int, int], dict[str, Any]] = {}
        try:
            with episodes_file.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    attempt_id = str(row.get("attempt_id", ""))
                    if not attempt_id or attempt_id not in completed_attempts:
                        continue
                    trajectory_idx = int(row.get("trajectory_idx", row.get("run_idx", 1)))
                    episode_idx = int(
                        row.get("episode_position", row.get("episode_idx", 0))
                    )
                    attempt_idx = int(row.get("attempt_idx", 1))
                    attempt_counts.setdefault(trajectory_idx, {})[episode_idx] = max(
                        attempt_counts.get(trajectory_idx, {}).get(episode_idx, 0),
                        attempt_idx,
                    )
                    key = (trajectory_idx, episode_idx)
                    old = latest.get(key)
                    if old is None or attempt_idx >= int(old.get("attempt_idx", 1)):
                        latest[key] = row
        except OSError as exc:
            print(f"Warning: could not reconcile episodes log: {exc}", file=sys.stderr)
            return

        changed = False
        for trajectory_idx in sorted(progress_by_trajectory):
            while progress_by_trajectory[trajectory_idx] < self.config.game.num_episodes:
                next_episode = progress_by_trajectory[trajectory_idx] + 1
                row = latest.get((trajectory_idx, next_episode))
                if row is None:
                    break
                expected_scenario = self._scenario_index(
                    trajectory_idx, next_episode
                )
                actual_scenario = int(row.get("scenario_idx", expected_scenario))
                if actual_scenario != expected_scenario:
                    raise RuntimeError(
                        "Completed log row conflicts with scenario plan for "
                        f"trajectory {trajectory_idx}, episode {next_episode}"
                    )
                trust_before = row.get("trust_before")
                trust_after = row.get("trust_after")
                if self._trust_schedule is not None:
                    self._validate_scheduled_trust_record(row)
                elif trust_before != trust_by_trajectory[trajectory_idx].snapshot():
                    raise RuntimeError(
                        "Completed log row has incompatible trust lineage for "
                        f"trajectory {trajectory_idx}, episode {next_episode}"
                    )
                if not isinstance(trust_after, dict) or not trust_after:
                    raise RuntimeError(
                        "Completed log row is missing trust_after for "
                        f"trajectory {trajectory_idx}, episode {next_episode}"
                    )
                if self._is_brs:
                    if row.get("brs_state_before") != self._serialize_evidence(self._brs_state[trajectory_idx]):
                        raise StateStoreError("Completed BRS row conflicts with evidence lineage")
                    self._restore_brs_state(trajectory_idx, row.get("brs_state_after"), trust_after)
                trust_by_trajectory[trajectory_idx] = TrustScores.from_snapshot(
                    trust_after
                )
                progress_by_trajectory[trajectory_idx] = next_episode
                changed = True

        if changed:
            self._write_checkpoint(
                progress_by_trajectory,
                trust_by_trajectory,
                attempt_counts,
            )

    def _validate_frozen_trust(self, snapshot: dict[str, dict[str, float]]) -> None:
        if self._trust_is_frozen:
            expected = TrustScores.initialize(
                self._scenarios[0].all_agent_ids(), self.config.trust.initial_score,
            ).snapshot()
            if snapshot != expected:
                raise RuntimeError("Frozen trust state must equal initial_score for every counterpart")

    def _build_config_fingerprint(self) -> dict[str, Any]:
        scenario_cycle_length = len(self._scenarios)
        prompt_config = self.config.prompts.model_dump()
        if self.config.prompts.prompt_version != "v5":
            prompt_config.pop("trust_decision_hint_level", None)
        fingerprint = {
            "supplement_prompt_protocol": SUPPLEMENT_PROMPT_PROTOCOL,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
            "seed_scheme_version": SEED_SCHEME_VERSION,
            "threshold_action_policy": THRESHOLD_ACTION_POLICY,
            "scenario_paths": list(self._scenario_paths),
            "scenario_hashes": list(self._scenario_hashes),
            "scenario_order_policy": getattr(
                self.config.game, "scenario_order_policy", "fixed"
            ),
            "scenario_order_template": self._scenario_orders(
                scenario_cycle_length
            ),
            "trust_mode": self.config.trust.mode,
            "trust_config": self.config.trust.model_dump(),
            "prompt_config": prompt_config,
            "malicious_config": self.config.malicious.model_dump(),
            "num_trajectories": self._trajectory_count(),
            "num_runs": self._trajectory_count(),
            "rounds_per_episode_multiplier": self.config.game.rounds_per_episode_multiplier,
            "base_seed": self.config.game.base_seed,
            "agent_ids": sorted(self._scenarios[0].all_agent_ids()),
            "agent_models": dict(getattr(self.config, "agent_models", {}) or {}),
            "effective_execution_backend": self._execution_backend(),
            "effective_dry_run": self._execution_backend() == "stub",
            "resolved_client_provenance": dict(self._agent_client_provenance),
            "llm_config": self.config.llm.model_dump(),
            "llm_temperature": self.config.llm.temperature,
            "llm_max_tokens": self.config.llm.max_tokens,
            "llm_run_entropy": self.config.llm.run_entropy,

        }
        if self.config.trust.brs is None:
            fingerprint["trust_config"].pop("brs", None)
        if self._trust_replay is None:
            fingerprint["trust_config"].pop("replay", None)
        else:
            fingerprint["trust_replay"] = self._trust_replay.metadata()
        if self._trust_random is None:
            fingerprint["trust_config"].pop("random_each_episode", None)
        else:
            fingerprint["trust_random"] = self._trust_random.metadata()
        if self._is_brs:
            fingerprint["brs_config"] = self._brs_metadata()
        if self._is_v4_protocol:
            fingerprint["trust_update_protocol"] = V4_TRUST_UPDATE_PROTOCOL
            fingerprint["trust_visibility_policy"] = (
                "disabled"
                if self.config.trust.mode == "none"
                else "v4_all_agents_from_episode1"
            )
        return fingerprint

    def _validate_checkpoint_fingerprint(self, fingerprint: dict[str, Any]) -> None:
        current = self._build_config_fingerprint()
        if fingerprint.get("supplement_prompt_protocol") != SUPPLEMENT_PROMPT_PROTOCOL:
            raise RuntimeError(
                "Supplement prompt protocol changed; use a new output directory "
                "instead of resuming results from a different experiment"
            )
        saved_hint = (fingerprint.get("prompt_config") or {}).get("trust_decision_hint_level")
        if saved_hint != self.config.prompts.trust_decision_hint_level:
            raise RuntimeError("S/S+P condition changed; use a new output directory")

        fingerprint = dict(fingerprint)
        saved_trust = dict(fingerprint.get("trust_config") or {})
        saved_trust.setdefault("is_frozen", False)
        if saved_trust.get("replay") is None:
            saved_trust.pop("replay", None)
        if not saved_trust.get("random_each_episode", False):
            saved_trust.pop("random_each_episode", None)
        fingerprint["trust_config"] = saved_trust
        if bool(saved_trust.get("random_each_episode")) != self._random_each_episode:
            raise RuntimeError("Config mismatch on resume: random trust changed; use a new output directory")
        if self._trust_random is not None and fingerprint.get("trust_random") != current["trust_random"]:
            raise RuntimeError("Config mismatch on resume: random trust generation settings changed")
        if bool(saved_trust.get("replay")) != (self._trust_replay is not None):
            raise RuntimeError("Config mismatch on resume: trust replay changed; use a new output directory")
        if self._trust_replay is not None and fingerprint.get("trust_replay") != current["trust_replay"]:
            raise RuntimeError("Config mismatch on resume: recorded trust schedule changed")
        if saved_trust["is_frozen"] != self._trust_is_frozen:
            raise RuntimeError(
                "Config mismatch on resume: trust.is_frozen changed. "
                "Use a new output directory for the other experimental condition."
            )
        if self._trust_is_frozen and saved_trust.get("initial_score") != self.config.trust.initial_score:
            raise RuntimeError("Config mismatch on resume: frozen trust.initial_score changed")
        mismatches = [
            key
            for key, current_value in current.items()
            if fingerprint.get(key) != current_value
        ]
        if not mismatches:
            return
        if self._trust_replay is not None:
            raise RuntimeError(f"Replay resume settings changed: {mismatches}")
        if self._trust_random is not None:
            raise RuntimeError(f"Random trust resume settings changed: {mismatches}")
        if self._is_brs or fingerprint.get("trust_mode") == "brs":
            raise StateStoreError(
                "BRS resume settings changed; saved feedback cannot be reused under a "
                f"different protocol, model or scenario: {mismatches}"
            )

        hard_backend_keys = {
            "effective_execution_backend",
            "effective_dry_run",
        }
        hard_backend_mismatches = hard_backend_keys.intersection(mismatches)
        if hard_backend_mismatches:
            raise RuntimeError(
                "Config mismatch on resume — execution backend changed between "
                "the checkpoint and current run "
                f"({sorted(hard_backend_mismatches)}). A dry-run/stub checkpoint "
                "can never be resumed with provider clients, or vice versa."
            )

        print(
            "\nWARNING: resumed config does not match the original run:",
            file=sys.stderr,
        )
        for key in mismatches:
            print(
                f"  {key}: checkpoint={fingerprint.get(key)!r} "
                f"current={current[key]!r}",
                file=sys.stderr,
            )

        if not self._resume_force:
            raise RuntimeError(
                "Config mismatch on resume — the current config differs from the "
                "checkpoint in ways that could invalidate experimental results. "
                "Fix the config, or pass --resume-force if the changes are "
                "intentional and harmless (e.g. extending num_episodes, changing "
                "parallel_workers or logging settings)."
            )
        print(
            "  --resume-force specified; continuing despite mismatches.\n",
            file=sys.stderr,
        )

    def _write_analysis_html(self) -> None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if not all((scripts_dir / name).is_file() for name in (
            "analysis_html.py", "qualitative_analysis_html.py",
        )):
            return
        try:
            from scripts.analysis_html import write_analysis_html
            out = write_analysis_html(run_dir)
            if getattr(self.logger, "console", True):
                print(f"  → Quantitative analysis written: {out}")
        except Exception as exc:
            print(f"Warning: quantitative analysis generation failed: {exc}", file=sys.stderr)
        try:
            from scripts.qualitative_analysis_html import write_qualitative_html
            out = write_qualitative_html(run_dir)
            if getattr(self.logger, "console", True):
                print(f"  → Qualitative analysis written: {out}")
        except Exception as exc:
            print(f"Warning: qualitative analysis generation failed: {exc}", file=sys.stderr)

    def _write_trajectory_replays(self) -> None:
        run_dir = getattr(self.logger, "run_dir", None)
        if run_dir is None:
            return
        if not (Path(__file__).resolve().parents[2] / "scripts" / "replay_html.py").is_file():
            return
        try:
            from scripts.replay_html import (
                render_all_episodes_html,
                render_replay_index_html,
                render_trajectory_html,
            )

            replay_dir = Path(run_dir) / "Replay"
            replay_dir.mkdir(parents=True, exist_ok=True)
            show_private = getattr(self.logger, "console_private", False)

            for pattern in ("replay_episode_*.html", "replay_trajectory_*.html"):
                for stale in replay_dir.glob(pattern):
                    stale.unlink()
            for trajectory_idx in range(1, self._trajectory_count() + 1):
                content = render_trajectory_html(
                    Path(run_dir),
                    trajectory_idx,
                    show_private=show_private,
                )
                output = replay_dir / f"replay_trajectory_{trajectory_idx}.html"
                output.write_text(content, encoding="utf-8")

            combined_output = replay_dir / "replay_all_trajectories.html"
            combined_output.write_text(
                render_all_episodes_html(
                    Path(run_dir),
                    show_private=show_private,
                ),
                encoding="utf-8",
            )
            index_output = replay_dir / "index.html"
            index_output.write_text(
                render_replay_index_html(Path(run_dir)),
                encoding="utf-8",
            )
            if getattr(self.logger, "console", True):
                print(
                    f"  → Trajectory replays written: {replay_dir} "
                    f"({self._trajectory_count()} trajectories + combined replay + index)"
                )
        except Exception as exc:
            print(f"Warning: trajectory replay generation failed: {exc}", file=sys.stderr)

    @staticmethod
    def _git_commit() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    @staticmethod
    def _scenario_hash(source_path: str) -> str:
        if not source_path:
            return ""
        try:
            content = Path(source_path).read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except OSError:
            return ""

    def _build_run_config(self) -> dict[str, Any]:
        g = self.config.game
        t = self.config.trust
        llm = self.config.llm
        num_trajectories = self._trajectory_count()
        return {
            "supplement_prompt_protocol": SUPPLEMENT_PROMPT_PROTOCOL,
            "malicious_strategy": "strategic",
            "design_version": LONGITUDINAL_DESIGN_VERSION,
            "seed_scheme_version": SEED_SCHEME_VERSION,
            "threshold_action_policy": THRESHOLD_ACTION_POLICY,
            "aggregation_unit": "trajectory",
            "num_episodes": g.num_episodes,
            "num_trajectories": num_trajectories,
            "num_runs": num_trajectories,
            "base_seed": g.base_seed,
            "trajectory_seeds": {
                str(trajectory_idx): derive_trajectory_seed(
                    g.base_seed, trajectory_idx
                )
                for trajectory_idx in range(1, num_trajectories + 1)
            },
            "scenario_order_policy": getattr(g, "scenario_order_policy", "fixed"),
            "scenario_orders": self._scenario_orders(),
            "scenario_plan_fingerprint": self._scenario_plan_fingerprint(
                self._scenario_orders()
            ),
            "parallel_workers": g.parallel_workers,
            "rounds_per_episode_multiplier": g.rounds_per_episode_multiplier,
            "trust_mode": t.mode,
            "trust_is_frozen": self._trust_is_frozen,
            "trust_condition_enabled": t.mode != "none",
            "trust_visible": t.mode != "none",
            "trust_visible_scope": "condition_level_compatibility_alias",
            "trust_visibility_policy": (
                "disabled"
                if t.mode == "none"
                else "v4_all_agents_from_episode1"
                if self._is_v4_protocol
                else "legacy_condition_enabled"
            ),
            "trust_initial_score": t.initial_score,
            "trust_decay": t.decay,
            "trust_metric_weights": dict(t.metric_weights),
            "trust_metric_weights_role": (
                "one_step_reference_only"
                if self._is_v4_protocol
                else "legacy_config_reference"
            ),
            "trust_update_protocol": (
                V4_TRUST_UPDATE_PROTOCOL if self._is_v4_protocol else "legacy"
            ),
            "trust_min_blocking_rejects": t.min_blocking_rejects,
            "trust_min_sample_rounds": t.min_sample_rounds,
            "oracle_unavailable_metric_default": self._oracle_unavailable_metric_default,
            "oracle_vote_inconsistency_source": "private_cover_utility_threshold",
            "oracle_state_basis": "actual_pre_episode_trust",
            "oracle_is_longitudinal_state": False,
            "prompt_config": self.config.prompts.model_dump(),
            "prompt_version": self.config.prompts.prompt_version,
            "trust_guidance": (
                "prescriptive" if self.config.prompts.trust_decision_hint_level == "prescriptive"
                and t.mode != "none" else "none"
            ),
            "trust_decision_hint_level": (
                self.config.prompts.trust_decision_hint_level
                if self.config.prompts.prompt_version == "v5"
                else "not_applicable"
            ),
            "malicious_config": self.config.malicious.model_dump(),
            "configured_agent_models": dict(
                getattr(self.config, "agent_models", {}) or {}
            ),
            "agent_models": dict(self._agent_model_names),
            "effective_execution_backend": self._execution_backend(),
            "effective_dry_run": self._execution_backend() == "stub",
            "resolved_client_provenance": dict(self._agent_client_provenance),
            "llm_config": llm.model_dump(),
            "llm_dry_run": llm.dry_run,
            "llm_api_version": llm.api_version,
            "llm_temperature": llm.temperature,
            "llm_max_tokens": llm.max_tokens,
            "llm_reasoning_model_prefixes": list(llm.reasoning_model_prefixes),
            "llm_run_entropy": llm.run_entropy,
            "llm_usage_prior_segments": dict(self._prior_llm_usage),
            "llm_usage_current_segment": self._aggregate_usage(),
            "llm_usage_cumulative": self._cumulative_usage(),
            "history_window_size": getattr(self.config.prompts, "history_window_size", None),
            "git_commit": self._git_commit(),

            "resume_start_episode": self._resume_start_episode,
            **({
                "brs_config": self._brs_metadata(),
                "trust_update_protocol": "brs_agent_feedback_v3",
                "oracle_trust_usage": "disabled",
                "oracle_vote_inconsistency_source": "disabled",
                "oracle_state_basis": "disabled",
            } if t.mode == "brs" and self._trust_schedule is None else {}),
            **({
                "trust_update_protocol": "frozen",
                "trust_configured_update_protocol": (
                    "brs_agent_feedback_v3" if t.mode == "brs"
                    else V4_TRUST_UPDATE_PROTOCOL if self._is_v4_protocol else "legacy"
                ),
                "oracle_trust_usage": "disabled",
                "oracle_vote_inconsistency_source": "disabled",
                "oracle_state_basis": "disabled",
            } if self._trust_is_frozen else {}),
            **({
                "trust_update_protocol": "recorded_trust_replay_v1",
                "trust_assessment_enabled": False,
                "trust_replay": self._trust_replay.metadata(),
                "oracle_trust_usage": "disabled",
                "oracle_vote_inconsistency_source": "disabled",
                "oracle_state_basis": "disabled",
            } if self._trust_replay is not None else {}),
            **({
                "trust_update_protocol": RANDOM_TRUST_PROTOCOL,
                "trust_random_each_episode": True,
                "trust_assessment_enabled": False,
                "trust_random": self._trust_random.metadata(),
                "trust_initial_score_role": "unused_random_from_episode1",
                "oracle_trust_usage": "disabled",
                "oracle_vote_inconsistency_source": "disabled",
                "oracle_state_basis": "disabled",
            } if self._trust_random is not None else {}),
        }

    def _build_scenario_metadata(self) -> dict[str, Any]:
        if not self._scenarios:
            return {}
        sc = self._scenarios[0]
        option_counts = {issue.id: len(issue.options) for issue in sc.issues}
        deal_space_size = 1
        for cnt in option_counts.values():
            deal_space_size *= cnt

        agent_types = {
            aid: getattr(cfg, "agent_type", "") or ""
            for aid, cfg in sc.agent_configs.items()
        }
        scenario_entries: list[dict[str, Any]] = []
        for scenario_idx, scenario in enumerate(self._scenarios, start=1):
            per_option_counts = {
                issue.id: len(issue.options) for issue in scenario.issues
            }
            per_deal_space = 1
            for count in per_option_counts.values():
                per_deal_space *= count
            scenario_entries.append({
                "scenario_idx": scenario_idx,
                "scenario_path": self._scenario_paths[scenario_idx - 1],
                "scenario_hash": self._scenario_hashes[scenario_idx - 1],
                "difficulty": getattr(scenario, "difficulty", ""),
                "n_agents": len(scenario.all_agent_ids()),
                "n_issues": len(scenario.issues),
                "issue_ids": [issue.id for issue in scenario.issues],
                "option_counts": per_option_counts,
                "deal_space_size": per_deal_space,
                "agent_types": {
                    aid: getattr(cfg, "agent_type", "") or ""
                    for aid, cfg in scenario.agent_configs.items()
                },
            })

        return {
            "scenario_hash": self._scenario_hashes[0],
            "num_scenarios": len(self._scenarios),
            "scenarios": scenario_entries,
            "n_agents": len(sc.all_agent_ids()),
            "n_issues": len(sc.issues),
            "issue_ids": [issue.id for issue in sc.issues],
            "option_counts": option_counts,
            "deal_space_size": deal_space_size,
            "agent_types": agent_types,
        }

    def _aggregate_usage(self) -> dict[str, int]:
        total = self._empty_usage()
        for client in self._agent_clients.values():
            if hasattr(client, "usage_summary"):
                usage = client.usage_summary()
                for key in total:
                    total[key] += int(usage.get(key, 0) or 0)
        return total

    @staticmethod
    def _empty_usage() -> dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }

    @classmethod
    def _normalise_usage(cls, usage: Any) -> dict[str, int]:
        normalised = cls._empty_usage()
        if not isinstance(usage, dict):
            return normalised
        for key in normalised:
            value = int(usage.get(key, 0) or 0)
            if value < 0:
                raise RuntimeError(f"Checkpoint has negative token usage for {key}")
            normalised[key] = value
        return normalised

    def _cumulative_usage(self) -> dict[str, int]:
        current = self._aggregate_usage()
        prior = self._brs_store.prior_usage(self._usage_segment) if self._is_brs else self._prior_llm_usage
        return {
            key: prior.get(key, 0) + current.get(key, 0)
            for key in self._empty_usage()
        }
