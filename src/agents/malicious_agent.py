from __future__ import annotations

from src.agents.honest_agent import HonestAgent
from src.utils.types import Deal, RoundRecord

class MaliciousAgent(HonestAgent):
    def propose_deal(
        self,
        history: list[RoundRecord],
        round_idx: int,
        total_rounds: int,
        is_opening: bool,
        initial_deal: str | None,
        display_names: dict[str, str],
    ) -> tuple[Deal, str, str]:
        return self._complete_proposer_turn(
            history=history,
            round_idx=round_idx,
            total_rounds=total_rounds,
            is_opening=is_opening,
            initial_deal=initial_deal,
            display_names=display_names,
        )

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

        if vote and not self.would_accept(deal):
            if hasattr(self.logger, "log_malicious_strategic_accept"):
                self.logger.log_malicious_strategic_accept(
                    agent_id=self.agent_id,
                    episode_idx=self.episode_idx,
                    run_idx=self.run_idx,
                    round_idx=round_idx + 1,
                    utility=self.compute_utility(deal),
                    threshold=self.config.min_threshold,
                    action_preserved=True,
                    disposition="logged_without_intervention",
                    **self._logger_identity(),
                )
        return vote, reason, scratchpad
