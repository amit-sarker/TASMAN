from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.utils.types import Deal, RoundRecord

THRESHOLD_ACTION_POLICY = "log_only_preserve_model_actions_v1"

class HonestAgent(BaseAgent):
    def propose_deal(
        self,
        history: list[RoundRecord],
        round_idx: int,
        total_rounds: int,
        is_opening: bool,
        initial_deal: str | None,
        display_names: dict[str, str],
    ) -> tuple[Deal, str, str]:
        deal, rationale, scratchpad = self._complete_proposer_turn(
            history=history,
            round_idx=round_idx,
            total_rounds=total_rounds,
            is_opening=is_opening,
            initial_deal=initial_deal,
            display_names=display_names,
        )
        if not self.would_accept(deal):
            if hasattr(self.logger, "log_vote_inconsistency"):
                self.logger.log_vote_inconsistency(
                    agent_id=self.agent_id,
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    round_idx=round_idx + 1,
                    utility=self.compute_utility(deal),
                    threshold=self.config.min_threshold,
                    context="proposer",
                    original_deal=dict(deal.choices),
                    original_rationale=rationale,
                    action_preserved=True,
                    disposition="logged_without_intervention",
                    **self._logger_identity(),
                )
        return deal, rationale, scratchpad

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
        vote, reason, scratchpad = self._complete_voter_turn(
            deal=deal,
            proposer_id=proposer_id,
            history=history,
            round_idx=round_idx,
            total_rounds=total_rounds,
            display_names=display_names,
            proposer_rationale=proposer_rationale,
        )

        round_num = round_idx + 1

        if vote and not self.would_accept(deal):
            if hasattr(self.logger, "log_vote_inconsistency"):
                self.logger.log_vote_inconsistency(
                    agent_id=self.agent_id,
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    round_idx=round_num,
                    utility=self.compute_utility(deal),
                    threshold=self.config.min_threshold,
                    context="voter",
                    original_reason=reason,
                    action_preserved=True,
                    disposition="logged_without_intervention",
                    **self._logger_identity(),
                )

        if not vote and self.would_accept(deal):
            if hasattr(self.logger, "log_above_threshold_reject"):
                deal_utility = self.compute_utility(deal)
                self.logger.log_above_threshold_reject(
                    agent_id=self.agent_id,
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    round_idx=round_num,
                    deal_utility=deal_utility,
                    min_threshold=self.config.min_threshold,
                    excess=deal_utility - self.config.min_threshold,
                    scratchpad=scratchpad,
                    vote_reason=reason,
                    **self._logger_identity(),
                )

        return vote, reason, scratchpad
