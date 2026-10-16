from __future__ import annotations

import hashlib
import random
from typing import Any

from src.game.deal import format_deal, parse_deal_string
from src.game.scenario import Scenario
from src.utils.seeding import LONGITUDINAL_DESIGN_VERSION
from src.utils.types import Deal, EpisodeRecord, RoundRecord, TrustScores, Vote

class Episode:
    def __init__(
        self,
        scenario: Scenario,
        agents: list[Any],
        episode_idx: int,
        trust_snapshot: dict[str, dict[str, float]],
        logger: Any,
        config: Any,
        run_idx: int = 1,
        rng: random.Random | None = None,
        run_seed: int = 0,
        *,
        trajectory_idx: int | None = None,
        trajectory_seed: int = 0,
        episode_seed: int | None = None,
        episode_position: int | None = None,
        scenario_idx: int = 0,
        scenario_path: str = "",
        scenario_hash: str | None = None,
        attempt_idx: int = 1,
        attempt_id: str = "",
    ) -> None:
        self.scenario = scenario
        self.agents = agents
        self.episode_idx = episode_idx
        self.episode_position = episode_position or episode_idx
        if self.episode_position != self.episode_idx:
            raise ValueError("episode_idx must equal episode_position")
        self.trajectory_idx = trajectory_idx or run_idx
        self.run_idx = self.trajectory_idx
        self.trajectory_seed = trajectory_seed
        self.trust_snapshot = trust_snapshot
        self.logger = logger
        self.config = config
        self.rng = rng
        self.episode_seed = episode_seed if episode_seed is not None else run_seed
        self.run_seed = self.episode_seed
        self.scenario_idx = scenario_idx
        self.scenario_path = scenario_path
        self.scenario_hash = (
            self._hash_scenario_path(scenario_path)
            if scenario_hash is None
            else scenario_hash
        )
        self.attempt_idx = attempt_idx
        self.negotiation_id = (
            f"t{self.trajectory_idx:03d}-e{self.episode_position:03d}"
        )
        self.attempt_id = attempt_id or (
            f"{self.negotiation_id}-a{self.attempt_idx:03d}"
        )
        self.history: list[RoundRecord] = []

    def _logger_identity(self) -> dict[str, Any]:
        return {
            "trajectory_idx": self.trajectory_idx,
            "trajectory_seed": self.trajectory_seed,
            "episode_position": self.episode_position,
            "scenario_idx": self.scenario_idx,
            "scenario_path": self.scenario_path,
            "scenario_hash": self.scenario_hash,
            "episode_seed": self.episode_seed,
            "negotiation_id": self.negotiation_id,
            "attempt_idx": self.attempt_idx,
            "attempt_id": self.attempt_id,
            "design_version": LONGITUDINAL_DESIGN_VERSION,
        }

    def run(self) -> EpisodeRecord:
        if self.rng is not None:
            self.rng.shuffle(self.agents)

        proposer_order = [agent.agent_id for agent in self.agents]

        n = len(self.agents)
        total_rounds = n * self.config.game.rounds_per_episode_multiplier
        min_non_proposer_accepts = n - 2
        display_names = {
            party.id: party.display_name
            for party in self.scenario.parties
        }
        malicious_agent_id = next(
            (agent.agent_id for agent in self.agents if agent.config.is_malicious),
            "",
        )
        def decision_uses_trust(agent: Any) -> bool:
            check = getattr(agent, "decision_uses_trust", None)
            return bool(check()) if callable(check) else bool(
                getattr(agent, "use_trust_scores", False)
            )

        thresholds_by_agent = {
            agent.agent_id: agent.config.min_threshold for agent in self.agents
        }

        agent_types = {
            agent.agent_id: getattr(agent.config, "agent_type", "") or ""
            for agent in self.agents
        }
        for agent in self.agents:
            agent.episode_idx = self.episode_idx
            agent.episode_position = self.episode_position
            agent.trajectory_idx = self.trajectory_idx
            agent.run_idx = self.run_idx
            agent.trajectory_seed = self.trajectory_seed
            agent.episode_seed = self.episode_seed
            agent.scenario_idx = self.scenario_idx
            agent.scenario_path = self.scenario_path
            agent.scenario_hash = self.scenario_hash
            agent.negotiation_id = self.negotiation_id
            agent.attempt_idx = self.attempt_idx
            agent.attempt_id = self.attempt_id

        trust_visibility_by_agent = {
            agent.agent_id: decision_uses_trust(agent)
            for agent in self.agents
        }
        trust_visible = any(trust_visibility_by_agent.values())
        trust_condition_enabled = self.config.trust.mode != "none"
        prompt_version = str(
            getattr(self.config.prompts, "prompt_version", "v1") or "v1"
        )
        trust_visibility_policy = (
            "disabled"
            if not trust_condition_enabled
            else "v4_all_agents_from_episode1"
            if prompt_version == "v4"
            else "legacy_condition_enabled"
        )

        for round_idx in range(total_rounds):
            round_num = round_idx + 1
            proposer = self.agents[round_idx % n]
            non_proposers = [a for a in self.agents if a.agent_id != proposer.agent_id]
            is_opening = round_idx == 0
            initial_deal = self._resolve_initial_deal(proposer) if is_opening else None

            deal, rationale, scratch_p = proposer.propose_deal(
                self.history,
                round_idx,
                total_rounds,
                is_opening,
                initial_deal,
                display_names,
            )

            deal_utilities: dict[str, int] = {
                agent.agent_id: agent.compute_utility(deal) for agent in self.agents
            }
            meets_threshold: dict[str, bool] = {
                agent.agent_id: deal_utilities[agent.agent_id] >= agent.config.min_threshold
                for agent in self.agents
            }
            proposer_util = deal_utilities[proposer.agent_id]
            proposer_threshold = thresholds_by_agent[proposer.agent_id]
            proposer_vote_type = (
                "proposer_self_below_threshold"
                if proposer_util < proposer_threshold
                else "rational_accept"
            )

            proposer_uses_trust = decision_uses_trust(proposer)
            proposer_trust_in_voters: dict[str, float | None] = (
                {
                    voter.agent_id: self.trust_snapshot.get(proposer.agent_id, {}).get(voter.agent_id)
                    for voter in non_proposers
                }
                if proposer_uses_trust else {}
            )

            votes: list[Vote] = []
            scratchpads = {proposer.agent_id: scratch_p}
            public_answers = {proposer.agent_id: rationale}
            for voter in non_proposers:
                vote_bool, reason, scratch_v = voter.vote_on_deal(
                    deal,
                    proposer.agent_id,
                    self.history,
                    round_idx,
                    total_rounds,
                    display_names,
                    rationale,
                )

                trust_in_prop: float | None = (
                    self.trust_snapshot.get(voter.agent_id, {}).get(proposer.agent_id)
                    if decision_uses_trust(voter) else None
                )

                v_util = deal_utilities[voter.agent_id]
                v_thresh = thresholds_by_agent[voter.agent_id]
                if vote_bool and v_util >= v_thresh:
                    vtype = "rational_accept"
                elif not vote_bool and v_util < v_thresh:
                    vtype = "mandatory_reject"
                elif not vote_bool and v_util >= v_thresh:
                    vtype = "strategic_reject"
                else:
                    vtype = "suspicious_accept"

                votes.append(Vote(
                    agent_id=voter.agent_id,
                    accept=vote_bool,
                    reason=reason,
                    trust_in_proposer=trust_in_prop,
                    vote_type=vtype,
                ))
                scratchpads[voter.agent_id] = scratch_v
                public_answers[voter.agent_id] = reason

            non_proposer_accepts = sum(1 for vote in votes if vote.accept)
            total_accepts = 1 + non_proposer_accepts
            passed = non_proposer_accepts >= min_non_proposer_accepts
            assert passed == (total_accepts >= n - 1)

            round_record = RoundRecord(
                round_idx=round_num,
                proposer_id=proposer.agent_id,
                deal=deal,
                rationale=rationale,
                votes=votes,
                proposer_implicit_accept=True,
                passed=passed,
                scratchpads=scratchpads,
                public_answers=public_answers,
            )
            self.history.append(round_record)
            if hasattr(self.logger, "log_round"):
                self.logger.log_round(
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    record=round_record,
                    total_accepts=total_accepts,
                    min_total_accepts=n - 1,
                    display_names=display_names,
                    deal_utilities_by_agent=deal_utilities,
                    meets_threshold_by_agent=meets_threshold,
                    proposer_utility=proposer_util,
                    proposer_vote_type=proposer_vote_type,
                    proposer_trust_in_voters=proposer_trust_in_voters,
                    trust_visible=trust_visible,
                    trust_visibility_by_agent=trust_visibility_by_agent,
                    prompt_version=prompt_version,
                    trust_visibility_policy=trust_visibility_policy,
                    **self._logger_identity(),
                )

            if passed:
                utilities = {
                    agent.agent_id: agent.compute_utility(deal)
                    for agent in self.agents
                }
                return EpisodeRecord(
                    episode_idx=self.episode_idx,
                    episode_position=self.episode_position,
                    trajectory_idx=self.trajectory_idx,
                    malicious_agent_id=malicious_agent_id,
                    agent_types=agent_types,
                    rounds=self.history,
                    success=True,
                    final_deal=deal,
                    utilities=utilities,
                    trust_before=self.trust_snapshot,
                    trust_mode=self.config.trust.mode,
                    trust_condition_enabled=trust_condition_enabled,
                    trust_visible=trust_visible,
                    trust_visibility_by_agent=trust_visibility_by_agent,
                    prompt_version=prompt_version,
                    trust_visibility_policy=trust_visibility_policy,
                    difficulty=self.scenario.difficulty,
                    rounds_to_deal=len(self.history),
                    thresholds_by_agent=thresholds_by_agent,
                    run_seed=self.run_seed,
                    trajectory_seed=self.trajectory_seed,
                    episode_seed=self.episode_seed,
                    scenario_idx=self.scenario_idx,
                    scenario_path=self.scenario_path,
                    scenario_hash=self.scenario_hash,
                    design_version=LONGITUDINAL_DESIGN_VERSION,
                    negotiation_id=self.negotiation_id,
                    attempt_idx=self.attempt_idx,
                    attempt_id=self.attempt_id,
                    proposer_order=proposer_order,
                    max_rounds=total_rounds,
                    n_agents=n,
                )

        utilities = {agent.agent_id: agent.config.min_threshold for agent in self.agents}
        return EpisodeRecord(
            episode_idx=self.episode_idx,
            episode_position=self.episode_position,
            trajectory_idx=self.trajectory_idx,
            malicious_agent_id=malicious_agent_id,
            agent_types=agent_types,
            rounds=self.history,
            success=False,
            final_deal=None,
            utilities=utilities,
            trust_before=self.trust_snapshot,
            trust_mode=self.config.trust.mode,
            trust_condition_enabled=trust_condition_enabled,
            trust_visible=trust_visible,
            trust_visibility_by_agent=trust_visibility_by_agent,
            prompt_version=prompt_version,
            trust_visibility_policy=trust_visibility_policy,
            difficulty=self.scenario.difficulty,
            rounds_to_deal=None,
            thresholds_by_agent=thresholds_by_agent,
            run_seed=self.run_seed,
            trajectory_seed=self.trajectory_seed,
            episode_seed=self.episode_seed,
            scenario_idx=self.scenario_idx,
            scenario_path=self.scenario_path,
            scenario_hash=self.scenario_hash,
            design_version=LONGITUDINAL_DESIGN_VERSION,
            negotiation_id=self.negotiation_id,
            attempt_idx=self.attempt_idx,
            attempt_id=self.attempt_id,
            proposer_order=proposer_order,
            max_rounds=total_rounds,
            n_agents=n,
        )

    @staticmethod
    def _hash_scenario_path(scenario_path: str) -> str:
        if not scenario_path:
            return ""
        try:
            with open(scenario_path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()[:16]
        except OSError:
            return ""

    def _resolve_initial_deal(self, proposer: Any) -> str | None:
        mode = self.config.prompts.opening_round.mode
        if mode == "model_generated":
            return None
        if mode == "fixed":
            fixed_deal = self.config.prompts.opening_round.fixed_deal
            if fixed_deal is None:
                raise ValueError("opening_round.fixed_deal must be set when mode='fixed'")
            parse_deal_string(fixed_deal, self.scenario.issues)
            return fixed_deal

        issue_order = [issue.id for issue in self.scenario.issues]
        if mode == "self_optimal":
            choices = {
                issue_id: max(proposer.config.scores[issue_id], key=proposer.config.scores[issue_id].get)
                for issue_id in issue_order
            }
            return format_deal(Deal(choices=choices), issue_order)

        if mode == "centroid":
            choices: dict[str, str] = {}
            for issue in self.scenario.issues:
                best_option = None
                best_total = None
                for option in issue.options:
                    total = sum(
                        self.scenario.agent_configs[agent_id].scores[issue.id][option.id]
                        for agent_id in self.scenario.all_agent_ids()
                    )
                    if best_total is None or total > best_total:
                        best_total = total
                        best_option = option.id
                assert best_option is not None
                choices[issue.id] = best_option
            return format_deal(Deal(choices=choices), issue_order)

        raise ValueError(f"Unsupported opening_round.mode '{mode}'")
