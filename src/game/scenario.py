from __future__ import annotations

from dataclasses import dataclass

from src.utils.types import AgentConfig, Deal, Issue, Party

@dataclass
class Scenario:
    project_name: str
    shared_description: str
    parties: list[Party]
    issues: list[Issue]
    agent_configs: dict[str, AgentConfig]

    _source_path: str = ""
    difficulty: str = "medium"

    def score_deal_for_agent(self, deal: Deal, agent_id: str) -> int:
        if agent_id not in self.agent_configs:
            raise KeyError(f"Unknown agent_id '{agent_id}'")

        agent = self.agent_configs[agent_id]
        total = 0
        for issue in self.issues:
            if issue.id not in deal.choices:
                raise ValueError(f"Deal is missing issue '{issue.id}'")
            option_id = deal.choices[issue.id]
            if option_id not in agent.scores[issue.id]:
                raise ValueError(
                    f"Illegal option '{option_id}' for issue '{issue.id}' in deal"
                )
            total += agent.scores[issue.id][option_id]
        return max(0, min(100, total))

    def all_agent_ids(self) -> list[str]:
        return [party.id for party in self.parties]

    def get_issue(self, issue_id: str) -> Issue:
        for issue in self.issues:
            if issue.id == issue_id:
                return issue
        raise KeyError(f"Unknown issue_id '{issue_id}'")
